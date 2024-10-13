import pandas as pd
import matplotlib.pyplot as plt
import re

# 定义一个函数来处理日志文件并提取F1分数
def extract_f1_scores_from_log(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()

    # 使用正则表达式来查找所有的F1分数，确保不匹配包含"Best Epoch"的行
    f1_scores = re.findall(r'Epoch: (\d+).*?Val F1: ([\.0-9]+)(?!.*Best Epoch)', log_data)

    # 转换成一个列表，包含(Epoch, F1 score)元组
    f1_scores = [(int(epoch), float(f1)) for epoch, f1 in f1_scores]

    return f1_scores

# 文件路径
file_path = 'results\\vast-lr=2e-05-bs=32-n_layers_fz=10-wiki=bert-base-n_gpus=1.txt'

# 提取F1分数
f1_scores = extract_f1_scores_from_log(file_path)

# 创建DataFrame
df_f1_scores = pd.DataFrame(f1_scores, columns=['Epoch', 'Val F1'])

# 显示DataFrame
df_even = df_f1_scores[df_f1_scores.index % 2 == 0]
df = df_even.reset_index(drop=True)
plt.figure(figsize=(10, 5))
plt.plot(df['Epoch'], df['Val F1'], marker='o', linestyle='-', color='b')
plt.title('Validation F1 Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Validation F1 Score')
plt.grid(True)
plt.xticks(df['Epoch'])  # 设置x轴刻度为Epoch值
plt.show()