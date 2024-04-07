# Author: Rose
# Time: 2024.04.07
# Personal version of "Project/0_2024.03.21-24/bert_sst2.py"

import os
import torch
import time
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, logging
from torch.optim import Adam
from torch.utils.data import dataloader, Dataset

logging.set_verbosity_error()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class BertSST2model(nn.Module):
    def __init__(self, class_size, pretrained_name='bert-base-chinese'):

    def forward(self, inputs):


class BertDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.datasetsize = len(dataset)


def save_pretrained(model, path):


def load_sentence_polarity(data_path, train_ratio=0.8):


def collate_fn(examples):


def main():

    # 初始化
    batch_size = 10
    num_epochs = 5
    check_step = 2
    data_path = "D:\\YCJH\\Project\\0_2024.03.21-24\\sst2_shuffled.tsv"
    train_ratio = 0.8
    learning_rate = 1e-5

    # 获取数据、分类类别总数
    train_data, test_data, categories = load_sentence_polarity(
        data_path, train_ratio)

    # 封装数据
    train_dataset = BertDataset(train_data)
    test_dataset = BertDataset(test_data)
    train_dataloader = dataloader(
        train_dataset, batch_size, collate_fn, shuffle=1)
    test_dataloader = dataloader(test_dataset, batch_size, collate_fn)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 加载模型
    pretrained_model_name = "D:\\YCJH\\Project\\0_2024.03.21-24\\bert-base-uncased"

    # 创建模型
    model = BertSST2model(len(categories), pretrained_model_name).to(device)

    # 加载预训练模型对应的tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    # Adam
    # optimizer = Adam(model.parameters(), learning_rate)

    # 定义损失函数
    CE_loss = nn.CrossEntropyLoss()

    # 训练过程
    model.train()
    for epoch_index in range(1, num_epochs+1):

        # 记录当前epoch的总loss
        total_loss = 0

        for batch_index in tqdm(train_dataloader, desc=f"Training Epoch {epoch_index}:"):


if __name__ == '__main__':
    main()
