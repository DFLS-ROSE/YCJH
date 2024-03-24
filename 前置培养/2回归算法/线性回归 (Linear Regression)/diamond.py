import pandas as pd    
import numpy as np    
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression    
from sklearn.metrics import mean_squared_error    
  
# 加载数据  
data = pd.read_csv("diamonds.csv")  

# 查看数据前5行  
print(data.head())  
  
# 定义我们的目标变量（价格）和特征变量（克拉重量，颜色等级，清晰度等级，切工等级）  
X = data[['carat', 'color', 'clarity', 'cut']]  
y = data['price']  
  
# 划分数据集为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# 创建线性回归模型并进行训练  
model = LinearRegression()  
model.fit(X_train, y_train)  
  
# 在测试集上进行预测  
y_pred = model.predict(X_test)  
  
# 计算均方误差（MSE）以评估模型性能  
mse = mean_squared_error(y_test, y_pred)  
print(f'Mean Squared Error: {mse}')