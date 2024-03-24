import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression, Ridge, Lasso  
from sklearn.metrics import mean_squared_error  
from sklearn.preprocessing import PolynomialFeatures  
from matplotlib import pyplot as plt  
  
# 生成训练数据和测试数据  
train_size = 20  
test_size = 12  
X_train = np.random.uniform(low=0, high=1.2, size=train_size)  
X_test = np.random.uniform(low=0.1, high=1.3, size=test_size)  
y_train = np.sin(X_train * 2 * np.pi) + np.random.normal(0, 0.2, train_size)  
y_test = np.sin(X_test * 2 * np.pi) + np.random.normal(0, 0.2, test_size)  
  
# 将训练数据和测试数据转换为DataFrame格式  
X_train = pd.DataFrame(X_train, columns=['Feature'])  
y_train = pd.DataFrame(y_train, columns=['Label'])  
X_test = pd.DataFrame(X_test, columns=['Feature'])  
y_test = pd.DataFrame(y_test, columns=['Label'])  
  
# 将训练数据分为训练集和验证集  
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  
  
# 创建 PolynomialFeatures 对象并设置多项式的阶数为 2  
poly_features = PolynomialFeatures(6)  
  
# 使用 PolynomialFeatures 对训练集和验证集进行特征转换  
X_train_poly = poly_features.fit_transform(X_train)  
X_val_poly = poly_features.transform(X_val)  
  
# 使用线性回归模型进行训练和预测  
model = LinearRegression()  
model.fit(X_train_poly, y_train)  
y_pred_poly = model.predict(X_val_poly)  
mse = mean_squared_error(y_val, y_pred_poly)  
print(f"MSE for Polynomial Regression: {mse}")  
  
# 绘制散点图和预测曲线图  
plt.scatter(X_val, y_val, color='blue', label='Validation Data')  
plt.plot(X_val, y_pred_poly, color='red', label='Polynomial Regression')  
plt.legend()  
plt.title('Scatter Plot and Prediction Curves')  
plt.show()