# Author: Rose
# Time: 2024.04.07
# Personal version of "Project/0_2024.03.21-24/bert_sst2.py"

import os, torch, time, numpy
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, logging
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score

logging.set_verbosity_error()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class BertSST2model(nn.Module):
    def __init__(self, class_size, pretrained_name='bert-base-chinese'):

        # 类继承固定格式
        super(BertSST2model, self).__init__()

        # 加载预训练模型
        self.bert = BertModel.from_pretrained(
            pretrained_name, return_dict=True)

        # 分类器：将768维的BERT输出映射到分类类别数
        self.classifier = nn.Linear(768, class_size)

    def forward(self, inputs):  # 前向传播

        # 输入数据
        inputs_ids = inputs['input_ids']
        inputs_tyi = inputs['token_type_ids']
        inputs_attn_mask = inputs['attention_mask']

        # 调用BERT模型得到输出
        outputs = self.bert(inputs_ids, inputs_tyi, inputs_attn_mask)

        # 将输出映射到分类类别数
        categories_num = self.classifier(outputs.pooler_output)
        return categories_num


class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_size = len(dataset)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return self.dataset[index]


def save_pretrained(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'pytorch_model.pth'))


def load_sentence_polarity(data_path, train_ratio=0.8): # 划分数据集和训练集, 类似于 sklearn 中的 train_test_split
    all_data = []
    categories = set() # 使用集合去重
    with open(data_path, 'r', encoding='utf-8') as f:
        for sample in f.readlines():
            polar, sent = sample.strip().split('\t')
            categories.add(polar)
            all_data.append((polar, sent))
    
    length = len(all_data)
    train_size = int(length * train_ratio)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]
    return train_data, test_data, categories

def __collate_fn(examples):
    inputs, targets = [], []
    for polar, sentence in examples:
        inputs.append(sentence)
        targets.append(int(polar))
    inputs = tokenizer(inputs, padding=True, truncation=True,
                       return_tensors='pt', max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets


# 初始化
# batch_size = 10
# num_epochs = 5
batch_size = 5
num_epochs = 1
check_step = 2
# data_path = "D:\\YCJH\\Project\\0_2024.03.21-24\\sst2_shuffled.tsv"
data_path = "D:\\YCJH\\Project\\0_2024.03.21-24\\sst2_shuffled_debug.tsv"
train_ratio = 0.8
learning_rate = 1e-5

# 获取数据、分类类别总数
train_data, test_data, categories = load_sentence_polarity(
    data_path, train_ratio)

# 封装数据
train_dataset = BertDataset(train_data)
test_dataset = BertDataset(test_data)
train_dataloader = DataLoader(
    train_dataset, batch_size, collate_fn=__collate_fn, shuffle=1)
test_dataloader = DataLoader(
    test_dataset, batch_size, collate_fn=__collate_fn)

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
optimizer = Adam(model.parameters(), learning_rate)

# 定义损失函数
CE_loss = nn.CrossEntropyLoss()

# 记录时间戳
timestamp = time.strftime("%m_%d_%H_%M", time.localtime())


# 训练过程
model.train()
for epoch_index in range(1, num_epochs+1):

    # 记录当前epoch的总loss
    total_loss = 0

    for train_batch_index in tqdm(train_dataloader, desc=f"Training Epoch {epoch_index}"):
        
        inputs, targets = [x.to(device) for x in train_batch_index]

        """
        等效于：
        for x in batch_index:
            inputs, targets = x.to(device)
        """

        # 清除现有梯度（防止梯度累积）
        optimizer.zero_grad()

        # 前向传播
        bert_output = model(inputs)

        # 计算损失
        loss = CE_loss(bert_output, targets)

        # 梯度反向传播并更新模型参数
        loss.backward()
        optimizer.step()

        # 统计总损失
        total_loss += loss.item()

    # 测试
    acc = 0

    predictions = []
    labels = []

    acc_debug = 0
    index = 1
    debug_bert_output = None
    debug_target = None
    # for test_batch_index in test_dataloader:
    for test_batch_index in tqdm(test_dataloader, desc=f"Testing"):
        # acc = 0
        inputs, targets = [x.to(device) for x in test_batch_index]
        # 输出预测张量
        bert_output = model(inputs)
        debug_bert_output = bert_output
        debug_target = targets
        # 累加结果
        acc += (bert_output.argmax(dim=1) == targets).sum().item()
        # .argmax(): 返回张量中最大值的位置, dim=1表示从第1维(行)开始索引

        # debug:
        predictions.extend(bert_output.argmax(dim=1).cpu().numpy())
        labels.extend(targets.cpu().numpy())
        # acc_debug = accuracy_score(targets.cpu().numpy(), bert_output.argmax(dim=1).cpu().numpy())
        # print(f"\n\n-------------------------DEBUG-INFO {index}-------------------------")
        # print(f"len(bert_output): {len(bert_output)}")
        # print(f"origin_acc: {acc}")
        # print(f"Acc: {acc/len(bert_output):.2f}")
        # print(f"Acc_from_sklearn: {acc_debug:.2f}")
        # print(f"Debug_bert_output: {debug_bert_output}")
        # print(f"Debug_target: {debug_target}")
        # print("-------------------------DEBUG-INFO END-------------------------\n\n")
        # index += 1
    
    # print(f"acc: {acc/len(test_dataloader.dataset):.2f}")
    print(f"acc_from_sklearn: {accuracy_score(labels, predictions):.2f}")

    if epoch_index % check_step == 0:
        # 保存模型
        checkpoint_dirname = "bert_ss2_" + timestamp
        os.makedirs(checkpoint_dirname, exist_ok=True)
        save_pretrained(
            model, checkpoint_dirname + '/checkpoints_{}'.format(epoch_index))

# END
