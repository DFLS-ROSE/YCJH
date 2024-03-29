# 一个基于Bert的简单文本分类代码

import torch  
from torch.utils.data import DataLoader, TensorDataset  
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup  
from sklearn.model_selection import train_test_split  
  
# 假设你有一个包含文本和对应标签的数据集，这里用一些虚拟数据来演示  
texts = ["我喜欢这个电影", "这本书很难读", "这首歌真好听", "我讨厌这个味道", "今天天气真好"]  
labels = [1, 0, 1, 0, 1]  # 假设1代表正面情感，0代表负面情感  
  
# 初始化BERT tokenizer和预训练模型  
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)  
  
# 将文本转换为BERT的输入格式  
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')  
input_ids = inputs['input_ids']  
attention_mask = inputs['attention_mask']  
labels = torch.tensor(labels)  
  
# 划分训练集和测试集  
input_ids_train, input_ids_test, attention_mask_train, attention_mask_test, labels_train, labels_test = train_test_split(input_ids, attention_mask, labels, test_size=0.2, random_state=42)  
  
# 创建PyTorch DataLoader  
train_dataset = TensorDataset(input_ids_train, attention_mask_train, labels_train)  
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)  
  
# 定义优化器和学习率调度器  
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  
epochs = 3  
total_steps = len(train_dataloader) * epochs  
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)  
  
# 训练模型  
model.train()  
for epoch in range(epochs):  
    for batch in train_dataloader:  
        batch = tuple(t.to(device) for t in batch)  
        input_ids, attention_mask, labels = batch  
          
        # 前向传播  
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  
        loss = outputs.loss  
        logits = outputs.logits  
          
        # 反向传播和优化  
        loss.backward()  
        optimizer.step()  
        scheduler.step()  
        optimizer.zero_grad()  
          
        if (step + 1) % 100 == 0:  
            print(f'Epoch [{epoch+1}/{epochs}], Step [{step+1}/{total_steps}], Loss: {loss.item():.4f}')  
  
# 在测试集上评估模型  
model.eval()  
with torch.no_grad():  
    correct = 0  
    total = 0  
    for batch in DataLoader(TensorDataset(input_ids_test, attention_mask_test, labels_test), batch_size=len(input_ids_test)):  
        batch = tuple(t.to(device) for t in batch)  
        input_ids, attention_mask, labels = batch  
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)  
        _, predicted = torch.max(outputs.logits, 1)  
        total += labels.size(0)  
        correct += (predicted == labels).sum().item()  
  
    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')