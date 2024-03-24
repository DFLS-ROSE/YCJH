import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression, Ridge, Lasso  
from sklearn.metrics import mean_squared_error,accuracy_score 
  
train_size = 20
test_size = 12

# 划分训练集和测试集  
X_train = np.random.uniform(low=0,high=1.2,size=train_size)
X_test = np.random.uniform(low=0.1,high=1.3,size=test_size)
y_train = np.sin(X_train*2*np.pi) + np.random.normal(0,0.2,train_size)
y_test = np.sin(X_test*2*np.pi) + np.random.normal(0,0.2,test_size)

X_train,X_test = X_train.reshape(-1,1),X_test.reshape(-1,1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# 普通最小二乘法  
model_ols = LinearRegression()  
model_ols.fit(X_train, y_train)  
y_pred_ols = model_ols.predict(X_test)  
mse_ols = mean_squared_error(y_test, y_pred_ols)  
print(f"MSE for Ordinary Least Squares: {mse_ols}")  
  
# 岭回归  
model_ridge = Ridge(alpha=1.0)  # 设置正则化强度  
model_ridge.fit(X_train, y_train)  
y_pred_ridge = model_ridge.predict(X_test)  
mse_ridge = mean_squared_error(y_test, y_pred_ridge)  
print(f"MSE for Ridge Regression: {mse_ridge}")  
  
# Lasso回归  
model_lasso = Lasso(alpha=0.1)  # 设置正则化强度  
model_lasso.fit(X_train, y_train)  
y_pred_lasso = model_lasso.predict(X_test)  
mse_lasso = mean_squared_error(y_test, y_pred_lasso)  
print(f"MSE for Lasso Regression: {mse_lasso}")

import matplotlib.pyplot as plt  
  
# 绘制训练数据散点图  
plt.scatter(X_train, y_train, color='blue', label='Training Data')  
  
# 绘制普通最小二乘法预测曲线  
plt.plot(X_test, y_pred_ols, color='red', label='OLS Prediction')  
  
# 绘制岭回归预测曲线  
plt.plot(X_test, y_pred_ridge, color='green', label='Ridge Prediction')  
  
# 绘制Lasso回归预测曲线  
plt.plot(X_test, y_pred_lasso, color='purple', label='Lasso Prediction')  
  
# 绘制测试数据散点图  
plt.scatter(X_test, y_test, color='black', label='Test Data')  
  
# 设置图例和标题  
plt.legend()  
plt.title('Scatter Plot and Prediction Curves')  
  
# 显示图像  
plt.show()