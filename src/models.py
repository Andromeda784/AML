import argparse
import sys
import torch
import random
import numpy as np
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
from utils import SupClusterConLoss, Discriminator, construct_hard_pairs, adversarial_loss
from description import ED_description, GE_description



class CrossEncoder(nn.Module):
    def __init__(self, args, num_labels, tkr, head='mlp', feat_dim=128):
        super(CrossEncoder, self).__init__()
        self.num_labels = num_labels
        self.encoder_type = args.encoder_type
        self.device = args.device
        # self.beta = args.beta
        self.alpha = args.alpha
        self.eta = args.eta
        self.gamma = args.gamma
        self.seed = args.seed

        label_description = ED_description if args.dataset == 'ED' else GE_description
        max_seq_len = 128 if args.dataset == 'go_emotion' else 64
        label_input_ids, label_attn_masks = [], []
        for label, description in label_description.items():
            label_tokens = tkr.tokenize(label)
            description_tokens = tkr.tokenize(description)

            tokens = label_tokens + ["<#>"] + description_tokens
            tokens = [tkr.cls_token] + tokens + [tkr.sep_token]
            attn_mask = [1] * len(tokens)
            if len(tokens) < max_seq_len:
                attn_mask += [0] * (max_seq_len - len(tokens))
                tokens += [tkr.pad_token] * (max_seq_len - len(tokens))

            input_ids = tkr.convert_tokens_to_ids(tokens)
            label_input_ids.append(input_ids)
            label_attn_masks.append(attn_mask)
        self.label_input_ids = torch.LongTensor(label_input_ids).to(args.device)
        self.label_attn_masks = torch.tensor(label_attn_masks).to(args.device)

        self.encoder = AutoModel.from_pretrained(args.encoder_type)
        self.encoder.resize_token_embeddings(len((tkr)))
        self.config = self.encoder.config
        self.label_embedding = nn.Embedding(num_labels, self.config.hidden_size)
        self.label_embedding_mat = nn.Parameter(torch.randn(num_labels, self.config.hidden_size,
                                                            requires_grad=True, device=self.device))
        self.wloss = WLoss([self.alpha, self.gamma])

        if head == 'linear':
            self.head = nn.Linear(self.config.hidden_size, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                # self.dropout = nn.Dropout(self.config.hidden_dropout_prob),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, feat_dim)
            )
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

        self.is_W_loss = False
        self.constrative_criterion = SupClusterConLoss(args.device,
                                                       args.temperature,
                                                       args.base_temperature,
                                                       0,
                                                       len(label_input_ids),
                                                       self.alpha,
                                                       self.gamma)
        # self.constrative_criterion = SupLoss(self.device)
        
        self.sim_threshold = getattr(args, 'sim_threshold', 0.5)
        self.topk = getattr(args, 'topk', 10)
        self.discriminator = Discriminator(feat_dim=feat_dim, 
                                         hidden_dim=getattr(args, 'disc_hidden_dim', 256),
                                         method=getattr(args, 'disc_method', 'mlp'))
        self.discriminator.to(self.device)

    def forward(self, input_ids, attention_mask, labels, return_adversarial_info=False):
        feats = self.head(self.encoder(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0, :])
        label_emb = self.encoder(input_ids=self.label_input_ids,
                                     attention_mask=self.label_attn_masks)[0][:, 0, :]
        feats_label = self.head(label_emb)

        logits = torch.matmul(feats, (feats_label).T)
        __loss = F.cross_entropy(logits, labels, reduction='mean')

        feats_norm = F.normalize(feats, dim=1)
        mask, logits_mask = self.sup_kmeans(feats_norm.detach().cpu().numpy(),
                                            labels.detach().cpu().numpy())

        # labels = torch.tensor(labels, device=self.device)
        # zero_mask_idxs = torch.nonzero(mask.sum(1) == 0).squeeze(1)
        zero_logits_mask_idxs = torch.nonzero(logits_mask.sum(1) == 0).squeeze(1)
        for idx in zero_logits_mask_idxs:
            tensors = torch.nonzero(~(labels == labels[idx])).squeeze(1)
            indices = torch.randperm(tensors.shape[0])
            neg_num = min(3, tensors.shape[0])
            logits_mask[idx, tensors[indices[:neg_num]]] = 1

        # loss += cate_loss(self.label_embedding())
        # W_loss = torch.tensor(0, device=self.device)
        # cnt = 0
        if self.is_W_loss:
            # self.constrative_criterion(feats.unsqueeze(1), labels)
            loss_, loss__, loss___ = self.constrative_criterion(feats, labels,
                                               label_emb, self.num_labels,
                                               mask, logits_mask, logits)
            _loss = self.wloss([loss___, loss__])
            # cnt += 3
        else:
            sys.exit()
        # loss_ += __loss
        
        total_loss = __loss + _loss
        
        return logits, total_loss

    def get_feats(self, input_ids, attention_mask):
        feats = self.encoder(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0, :]
        feats_norm = self.head(feats)

        return feats_norm

    def sup_kmeans(self, features, labels):
        c, l = len(np.unique(labels)), features.shape[0]
        #
        # min_J, opt_M = 1e6, 0
        # for M in tqdm(range(c, c + 10)):
        #     kmeans = KMeans(n_clusters=M, init='k-means++', random_state=0)
        #     kmeans.fit_predict(features)
        #
        #     purity = 0
        #     for cluster_id in range(M):
        #         indices = np.where(kmeans.labels_ == cluster_id)
        #         purity += np.max(counts)
        #     purity = (l - purity) / l
        #     J = purity + np.sqrt((M - c) / l)

        opt_M = c
        kmeans = KMeans(n_clusters=opt_M, init='k-means++', random_state=self.seed)
        kmeans.fit_predict(features)
        cluster_labels = kmeans.labels_
        # P = np.zeros((opt_M, self.label_num)
        # for cluster_id in range(opt_M):
        # indices = np.where(cluster_labels == cluster_id)
        #     unique_labels, counts = np.unique(labels[indices], return_counts=True)
        # P = P / np.sum(P, axis=1, keepdims=True)
        # purity = np.max(P, axis=1)

        mask, logits_mask = torch.zeros(l, l, device=self.device), \
            torch.zeros(l, l, device=self.device)
        for i in range(l):
            for j in range(i + 1, l):
                if cluster_labels[i] == cluster_labels[j] and labels[i] != labels[j]:
                    logits_mask[i, j] = logits_mask[j, i] = 1
                elif cluster_labels[i] != cluster_labels[j] and labels[i] == labels[j]:
                    logits_mask[i, j] = logits_mask[j, i] = 1
                    mask[i, j] = mask[j, i] = 1

        # labels = torch.tensor(labels, device=self.device)
        # # zero_mask_idxs = torch.nonzero(mask.sum(1) == 0).squeeze(1)
        #
        # for idx in torch.nonzero(logits_mask.sum(1) == 0).squeeze(1):
        #     tensors = torch.nonzero(~(labels == labels[idx])).squeeze(1)
        #     indices = torch.randperm(tensors.shape[0])

        return mask, logits_mask


class WLoss(nn.Module):
    def __init__(self, w):
        super(WLoss, self).__init__()
        self.w = w

    def forward(self, inputs):
        tl = 0
        for i in range(len(self.w)):
            tl += self.w[i] * inputs[i]
        return tl