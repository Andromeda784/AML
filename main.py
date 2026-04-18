import argparse
import sys, os
import torch
import random
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import wandb
import datetime
from datasets import *
from models import CrossEncoder
from utils import EarlyStopping, construct_hard_pairs, adversarial_loss
# import plotly.express as px
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")


class ResultLogger:
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.results = {
            'train': [],
            'eval': [],
            'test': None
        }
        self.step_losses = {
            'disc_loss': [],
            'enc_adv_loss': [],
            'step': []
        }
        self.global_step = 0
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f'Training started at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write('=' * 80 + '\n\n')
    
    def log(self, message, print_to_console=True):
        if print_to_console:
            print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
    
    def save_results(self):
        if not self.log_file:
            return
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write('\n' + '=' * 80 + '\n')
            f.write('Training Results Summary\n')
            f.write('=' * 80 + '\n\n')
            
            if self.results['train']:
                f.write('Training Process:\n')
                f.write('-' * 80 + '\n')
                f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Train F1':<12} {'Train Macro F1':<15} "
                       f"{'Eval Loss':<12} {'Eval Acc':<12} {'Eval F1':<12} {'Eval Macro F1':<15}\n")
                f.write('-' * 80 + '\n')
                for epoch_data in self.results['train']:
                    epoch = epoch_data.get('epoch', '')
                    train = epoch_data.get('train', {})
                    eval_data = epoch_data.get('eval', {})
                    
                    def format_value(val, default='N/A', fmt='.4f'):
                        if val == 'N/A' or val is None:
                            return f"{default:<12}"
                        try:
                            return f"{val:{fmt}}"
                        except:
                            return f"{default:<12}"
                    
                    f.write(f"{epoch:<8} "
                           f"{format_value(train.get('loss'), 'N/A', '<12.4f')} "
                           f"{format_value(train.get('acc'), 'N/A', '<12.4f')} "
                           f"{format_value(train.get('f1'), 'N/A', '<12.4f')} "
                           f"{format_value(train.get('marco_f1'), 'N/A', '<15.4f')} "
                           f"{format_value(eval_data.get('loss'), 'N/A', '<12.4f')} "
                           f"{format_value(eval_data.get('acc'), 'N/A', '<12.4f')} "
                           f"{format_value(eval_data.get('f1'), 'N/A', '<12.4f')} "
                           f"{format_value(eval_data.get('marco_f1'), 'N/A', '<15.4f')}\n")
                f.write('\n')
            
            if self.results['test']:
                f.write('Test Results:\n')
                f.write('-' * 80 + '\n')
                test = self.results['test']
                test_loss = test.get('loss', 'N/A')
                test_acc = test.get('acc', 'N/A')
                test_f1 = test.get('f1', 'N/A')
                test_marco_f1 = test.get('marco_f1', 'N/A')
                
                f.write(f"Loss: {test_loss:.4f}\n" if isinstance(test_loss, (int, float)) else f"Loss: {test_loss}\n")
                f.write(f"Accuracy: {test_acc:.4f}\n" if isinstance(test_acc, (int, float)) else f"Accuracy: {test_acc}\n")
                f.write(f"Weighted F1: {test_f1:.4f}\n" if isinstance(test_f1, (int, float)) else f"Weighted F1: {test_f1}\n")
                f.write(f"Macro F1: {test_marco_f1:.4f}\n" if isinstance(test_marco_f1, (int, float)) else f"Macro F1: {test_marco_f1}\n")
                if isinstance(test_acc, (int, float)) and isinstance(test_f1, (int, float)):
                    f.write(f"Acc + F1: {test_acc + test_f1:.4f}\n")
                f.write('\n')
            
            f.write(f'Training ended at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        
        if self.step_losses['disc_loss'] or self.step_losses['enc_adv_loss']:
            step_loss_file = self.log_file.replace('.txt', '_step_losses.json') if self.log_file else None
            if step_loss_file:
                import json
                with open(step_loss_file, 'w', encoding='utf-8') as f:
                    json.dump(self.step_losses, f, indent=2, ensure_ascii=False)
        
        print(f'\nAll results saved to: {self.log_file}')


def train(model, train_loader, opt, schedule, disc_opt=None, logger=None):
    model.train()
    model.discriminator.train()

    total_loss, all_preds, all_labels = 0, [], []
    total_adv_loss_disc = 0
    total_adv_loss_enc = 0
    batch_count_disc = 0
    batch_count_enc = 0
    cnt_1, cnt_2 = 0, 0
    
    for input_ids, attention_mask, labels, _ in tqdm(train_loader):
        input_ids, attention_mask, labels =             input_ids.to(args.device), attention_mask.to(args.device), \
            labels.to(args.device)
        
        feats = model.get_feats(input_ids, attention_mask)
        feats_norm = F.normalize(feats, dim=1)
        
        P_hard_pos, P_hard_neg, _ = construct_hard_pairs(
            feats_norm, labels, model.sim_threshold, args.device,
            use_topk=True, topk=model.topk
        )
        
        if disc_opt is not None and len(P_hard_pos) + len(P_hard_neg) > 0:
            disc_opt.zero_grad()
            adv_loss_disc_before = adversarial_loss(
                model.discriminator, feats_norm.detach(), P_hard_pos, P_hard_neg, args.device
            )
            (adv_loss_disc_before).backward()
            disc_opt.step()
            
            with torch.no_grad():
                adv_loss_disc_after = adversarial_loss(
                    model.discriminator, feats_norm.detach(), P_hard_pos, P_hard_neg, args.device
                )
            disc_loss_value = adv_loss_disc_after.item()
            total_adv_loss_disc += disc_loss_value
            batch_count_disc += 1
            
            if logger:
                logger.step_losses['disc_loss'].append(disc_loss_value)
                logger.step_losses['step'].append(logger.global_step)
                logger.global_step += 1
        
        opt.zero_grad()
        logits, cls_loss = model(input_ids, attention_mask, labels, return_adversarial_info=False)
        adv_loss_enc_before = adversarial_loss(
            model.discriminator, feats_norm, P_hard_pos, P_hard_neg, args.device
        )
        total_loss_combined = cls_loss + args.lambda_adv * adv_loss_enc_before
        total_loss_combined.backward()
        opt.step()
        schedule.step()
        
        with torch.no_grad():
            feats_enc_after = model.get_feats(input_ids, attention_mask)
            feats_norm_enc_after = F.normalize(feats_enc_after, dim=1)
            P_hard_pos_after, P_hard_neg_after, _ = construct_hard_pairs(
                feats_norm_enc_after, labels, model.sim_threshold, args.device,
                use_topk=True, topk=model.topk
            )
            adv_loss_enc_after = adversarial_loss(
                model.discriminator, feats_norm_enc_after, P_hard_pos_after, P_hard_neg_after, args.device
            )
        
        enc_adv_loss_value = adv_loss_enc_after.item()
        total_loss += cls_loss.item()
        total_adv_loss_enc += enc_adv_loss_value
        batch_count_enc += 1
        
        if logger:
            logger.step_losses['enc_adv_loss'].append(enc_adv_loss_value)
            if len(logger.step_losses['step']) < len(logger.step_losses['enc_adv_loss']):
                logger.step_losses['step'].append(logger.global_step)
                logger.global_step += 1

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    train_acc = accuracy_score(all_labels, all_preds)
    train_weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    marco_f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(train_loader)
    
    avg_adv_loss_disc = total_adv_loss_disc / batch_count_disc if batch_count_disc > 0 else 0.0
    avg_adv_loss_enc = total_adv_loss_enc / batch_count_enc if batch_count_enc > 0 else 0.0
    
    loss_str = f'Train => loss : {avg_loss:.04f}'
    if avg_adv_loss_disc > 0:
        loss_str += f', disc_loss : {avg_adv_loss_disc:.04f}'
    if avg_adv_loss_enc > 0:
        loss_str += f', enc_adv_loss : {avg_adv_loss_enc:.04f}'
    loss_str += f', acc : {train_acc:.04f}, f1 : {train_weighted_f1:.04f}, marco_f1 : {marco_f1:.04f}'
    
    if logger:
        logger.log(loss_str)
    else:
        print(loss_str)

    return train_acc, train_weighted_f1, marco_f1, avg_loss, avg_adv_loss_disc, avg_adv_loss_enc


def eval_or_test(model, loader, mode, logger=None):
    model.eval()

    all_logits = []
    total_loss,all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels, texts in loader:
            input_ids, attention_mask, labels = input_ids.to(args.device), attention_mask.to(args.device), \
                labels.to(args.device)
            logits, loss = model(input_ids, attention_mask, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            pred_ids = preds.detach().cpu().tolist()
            label_ids = labels.detach().cpu().tolist()
            all_preds.extend(pred_ids)
            all_labels.extend(label_ids)
            all_logits.extend(logits.detach().cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    marco_f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(loader)
    
    result_str = mode + f' => loss : {avg_loss:.04f}, ' \
                 f'acc : {acc:.04f}, ' \
                 f'weighted_f1 : {weighted_f1:.04f}, ' \
                 f'marco_f1 : {marco_f1:.04f}'
    
    if logger:
        logger.log(result_str)
    else:
        print(result_str)

    return acc, weighted_f1, marco_f1, avg_loss


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def main():
    # args.lr = float("{:.5e}".format(wandb.config.lr))
    # args.epochs = wandb.config.epochs
    # # args.weight_decay = float("{:.4e}".format(wandb.config.weight_decay))
    # # args.beta = wandb.config.beta
    # args.temperature = float("{:.5e}".format(wandb.config.temperature))
    # args.gamma = float("{:.5e}".format(wandb.config.gamma))

    log_file = getattr(args, 'result_file', f'./results/{args.dataset}_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = ResultLogger(log_file)
    
    logger.log('=' * 80)
    logger.log('Training Configuration:')
    logger.log('=' * 80)
    logger.log(f'Dataset: {args.dataset}')
    logger.log(f'Batch size: {args.batch_size}')
    logger.log(f'Epochs: {args.epochs}')
    logger.log(f'Learning rate: {args.lr}')
    logger.log(f'Weight decay: {args.weight_decay}')
    logger.log(f'Encoder type: {args.encoder_type}')
    logger.log(f'Device: {args.device}')
    logger.log(f'Adversarial training: True')
    logger.log(f'Adversarial loss weight: {args.lambda_adv}')
    logger.log(f'Hard pairs method: Top-{getattr(args, "topk", 10)}')
    logger.log('=' * 80)
    logger.log('')

    label2idx, idx2label = get_dicts(args.dataset)
    args.idx2label = idx2label
    num_classes = len(idx2label.items())
    class_names = [v for k, v in sorted(idx2label.items(), key=lambda item: item[0])]

    tkr = AutoTokenizer.from_pretrained(args.encoder_type)
    special_tokens_dict = {'additional_special_tokens': ["<#>"]}
    tkr.add_special_tokens(special_tokens_dict)

    trainset = emotion_DataSet('train', tkr)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=trainset.collate)
    validset = emotion_DataSet('valid', tkr)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, collate_fn=trainset.collate)
    testset = emotion_DataSet('test', tkr)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=trainset.collate)

    model = CrossEncoder(args, num_classes, tkr)
    model.to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # , no_deprecation_warning=True
    schedule = get_linear_schedule_with_warmup(opt, num_warmup_steps=0,  # len(train_loader) * 3
                                               num_training_steps=len(train_loader) * args.epochs)
    
    disc_opt = AdamW(model.discriminator.parameters(), 
                    lr=getattr(args, 'disc_lr', args.lr * 0.1), 
                    weight_decay=args.weight_decay)
    logger.log(f"Adversarial training enabled, discriminator learning rate: {getattr(args, 'disc_lr', args.lr * 0.1)}")
    
    model_suffix = '_adv'
    model_path = f'../mlp{args.device_id}{args.dataset}{model_suffix}.pt'
    early_stopping = EarlyStopping(patience=args.patience, type='acc', path=model_path)
    logger.log(f"Model will be saved to: {model_path}")

    for epoch in range(args.epochs):
        logger.log(f'----------- Epoch {epoch + 1} -------------')

        if epoch + 1 > 0:   # :
            model.is_W_loss = True

        train_result = train(model, train_loader, opt, schedule, disc_opt, logger)
        train_acc, train_f1, train_marco_f1, train_loss, disc_loss, enc_adv_loss = train_result
        
        eval_acc, eval_f1, eval_marco_f1, eval_loss = eval_or_test(model, valid_loader, 'Eval', logger)
        
        epoch_result = {
            'epoch': epoch + 1,
            'train': {
                'loss': train_loss,
                'acc': train_acc,
                'f1': train_f1,
                'marco_f1': train_marco_f1
            },
            'eval': {
                'loss': eval_loss,
                'acc': eval_acc,
                'f1': eval_f1,
                'marco_f1': eval_marco_f1
            }
        }
        
        epoch_result['train']['disc_loss'] = disc_loss
        epoch_result['train']['enc_adv_loss'] = enc_adv_loss
        
        logger.results['train'].append(epoch_result)
        
        # wandb.log({
        #     'train_ce_acc': train_acc,
        #     'train_ce_f1': train_f1,
        #     'train_marco_f1': train_marco_f1,
        #     'eval_ce_acc': eval_acc,
        #     'eval_ce_f1': eval_f1,
        #     'eval_marco_f1': eval_marco_f1
        # })

        early_stopping(eval_acc + eval_f1, model)
        if early_stopping.early_stop:
            logger.log(f"+++ early stop at epoch {epoch + 1 - args.patience} +++")
            break

    model_suffix = '_adv'
    checkpoint_path = f'../mlp{args.device_id}{args.dataset}{model_suffix}.pt'
    
    try:
        saved_state_dict = torch.load(checkpoint_path, map_location=args.device)
        missing_keys, unexpected_keys = model.load_state_dict(saved_state_dict, strict=False)
        if missing_keys:
            logger.log(f"Warning: Keys not loaded: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Warning: Keys not loaded: {missing_keys}")
        if unexpected_keys:
            logger.log(f"Warning: Keys ignored: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Warning: Keys ignored: {unexpected_keys}")
    except Exception as e:
        logger.log(f"Failed to load model: {e}")
        raise
    
    logger.log(f'----------- Test -------------')
    test_acc, test_f1, test_marco_f1, test_loss = eval_or_test(model, test_loader, 'Test', logger)
    test_result_str = f'test_acc_f1: {test_acc + test_f1:.4f}'
    logger.log(test_result_str)
    
    logger.results['test'] = {
        'loss': test_loss,
        'acc': test_acc,
        'f1': test_f1,
        'marco_f1': test_marco_f1
    }
    
    logger.save_results()
    
    # wandb.log({
    #     'test_acc': test_acc,
    #     'test_f1': test_f1,
    #     'test_marco_f1': test_marco_f1,
    # })

    # postfix = str(args.lr) + '_' + str(args.ce_lr) + '_' + str(args.temperature) \
    #           + '_' + str(args.scl_epochs) + '_' + str(args.ce_epochs)
    # postfix = '\_cluster'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ED')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--encoder_type', type=str, default='roberta-base')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=4)
    # parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--base_temperature', type=float, default=0.07)
    parser.add_argument('--device_id', type=str, default="0")
    parser.add_argument('--error_print_limit', type=int, default=20)
    parser.add_argument('--error_csv_prefix', type=str, default='./misclassified')
    parser.add_argument('--result_file', type=str, default=None, help='Result file save path (auto-generated by default)')
    parser.add_argument('--topk', type=int, default=10, help='Top-k number for hard pairs construction')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Adversarial loss weight λ')
    parser.add_argument('--disc_lr', type=float, default=None, help='Discriminator learning rate (default: 0.1x main lr)')
    parser.add_argument('--disc_hidden_dim', type=int, default=256, help='Discriminator hidden dimension')
    parser.add_argument('--disc_method', type=str, default='mlp', choices=['mlp', 'metric'], help='Discriminator implementation method')

    args = parser.parse_args()
    
    if args.disc_lr is None:
        args.disc_lr = args.lr * 0.1
    args.device = torch.device("cuda:" + args.device_id if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main()

    # sweep_configuration = {
    #     'method': 'bayes',
    # }
    # wandb.agent(sweep_id, function=main)