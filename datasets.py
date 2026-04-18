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
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import wandb
import datetime


all_dataset_list = ['go_emotion', 'ED']


def get_dicts(dataset):
    if dataset == 'ED':
        label2idx = {'sad': 0, 'trusting': 1, 'terrified': 2, 'caring': 3, 'disappointed': 4,
                     'faithful': 5, 'joyful': 6, 'jealous': 7, 'disgusted': 8, 'surprised': 9,
                     'ashamed': 10, 'afraid': 11, 'impressed': 12, 'sentimental': 13,
                     'devastated': 14, 'excited': 15, 'anticipating': 16, 'annoyed': 17, 'anxious': 18,
                     'furious': 19, 'content': 20, 'lonely': 21, 'angry': 22, 'confident': 23,
                     'apprehensive': 24, 'guilty': 25, 'embarrassed': 26, 'grateful': 27,
                     'hopeful': 28, 'proud': 29, 'prepared': 30, 'nostalgic': 31}

    elif dataset == 'go_emotion':
        label2idx = {'admiration': 0, 'amusement': 1, 'anger': 2,
                     'annoyance': 3, 'approval': 4, 'caring': 5,
                     'confusion': 6, 'curiosity': 7, 'desire': 8,
                     'disappointment': 9, 'disapproval': 10, 'disgust': 11,
                     'embarrassment': 12, 'excitement': 13, 'fear': 14,
                     'gratitude': 15, 'grief': 16, 'joy': 17,
                     'love': 18, 'nervousness': 19, 'optimism': 20,
                     'pride': 21, 'realization': 22, 'relief': 23,
                     'remorse': 24, 'sadness': 25, 'surprise': 26}

    idx2label = {v: k for k, v in label2idx.items()}
    return label2idx, idx2label


class emotion_DataSet(Dataset):
    def __init__(self, mode, tkr, dataset='go_emotion'):
        super().__init__()
        assert mode in ['train', 'valid', 'test']
        df = pd.read_csv(f'../data/{dataset}/{mode}.csv')

        self.text, self.label = df.text, df.label
        self.tkr = tkr

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]

    def collate(self, batch):
        text = [t for t, _ in batch]
        encode = self.tkr(text, padding='longest', truncation=True, max_length=200, return_tensors='pt')
        label = [l for _, l in batch]
        label_tensor = torch.tensor(label)
        return encode['input_ids'], encode['attention_mask'], label_tensor, text

if __name__ == '__main__':

    pass
