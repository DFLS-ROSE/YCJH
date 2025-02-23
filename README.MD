# 英才计划-文本立场检测研究

## 项目简介

本项目围绕零样本文本立场检测展开，基于深度学习方法实现跨主题立场分类。研究内容包含：

- 基于Wikipedia知识增强的立场检测模型
- 结合常识知识图谱的立场推理框架
- 融合情感特征的迁移学习方法

## 环境配置
### 创建conda环境
conda create -n stance python=3.8
conda activate stance

### 安装核心依赖
pip install torch==1.10.0 transformers==4.1.1 pandas matplotlib
pip install torch-scatter pytorch-geometric spacy gensim

## 项目结构
YCJH/
├── Project/
│   ├── 2_2024.07.29-08.04/SentKB-main  # 知识增强模型
│   ├── 2_2024.08.12-08.18/SentiX       # 情感增强模型
│   └── 2_2024.10.20-27/                # 实验结果与可视化
│       ├── draw.py                      # 指标可视化脚本
│       └── wiki-enhanced-stance-detection/ # 最终模型
└── data/
    ├── VAST/                           # 基准数据集
    └── ConceptNet/                      # 常识知识图谱

## 核心方法

1. 知识增强（wiki-enhanced）：
from models import KnowledgeEnhancedModel
model = KnowledgeEnhancedModel(bert_model='bert-base', kg_dim=300)

2. 情感融合（SentiX）：
from models.sentix import SentimentAwareModel
model = SentimentAwareModel(sentiment_dim=100)

## 实验结果
运行可视化脚本：
cd 2_2024.10.20-27
python draw.py

生成图表示例见 f1_scores.png

## 参考文献
1. Allaway et al. 《Zero-Shot Stance Detection》 (VAST数据集)
2. 浙江大学《Exploiting Sentiment and Common Sense for Zero-shot Stance Detection》
3. USC《Infusing Knowledge from Wikipedia to Enhance Stance Detection》
