from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  
import pandas as pd  
  
# 读取数据  
data = pd.read_csv('D:\\学习资料\\英才计划\\前置培养\\2线性回归 (Linear Regression)\\diamonds.csv')  
  
from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()  
def label_encoder(data):  
    return le.fit_transform(data)  
def transform_columns(data):  
    for column in data.columns:  
        data[column] = label_encoder(data[column].values)  
    return data  
data = transform_columns(data) 
# 定义特征和目标变量  
X = data[['carat', 'color', 'clarity', 'cut']]  
y = data['price']  
  
# 划分数据集为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# 创建线性回归模型  
model = LinearRegression()  
  
# 训练模型  
model.fit(X_train, y_train)  
  
# 在测试集上进行预测  
y_pred = model.predict(X_test)  
  
# 计算均方误差（MSE）和R²得分  
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  
  
# 打印拟合结果和评估指标  
print("均方误差（MSE）:", mse)  
print("R²得分:", r2)