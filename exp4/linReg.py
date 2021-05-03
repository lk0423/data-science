# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import dot
 
iris = pd.read_csv('iris.csv')
# 拟合线性模型： Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width

#print(iris)
# 矩阵解法
temp = iris.iloc[:,2:5]
temp['x0'] = 1
X = temp.iloc[:,[3,0,1,2]]
Y = iris.iloc[:,1]

theta_m = dot(dot(inv(dot(X.T, X)), X.T), Y) # theta = (X'X)^(-1)X'Y
print("矩阵解法：")
print("权重系数：",theta_m)


#批量梯度下降法

theta_g= np.zeros(4) #初始化theta
alpha = 0.01
#temp = np.zeros(4)
X0 = X.iloc[:, 0]
X1 = X.iloc[:, 1]
X2 = X.iloc[:, 2]
X3 = X.iloc[:, 3]

IterMax=800
J = pd.Series(np.arange(IterMax, dtype = float))
for i in range(IterMax):
#更新theta
    tmp=Y-dot(X, theta_g)
    theta_g[0] = theta_g[0] + alpha*np.sum(tmp*X0)/150.
    theta_g[1] = theta_g[1] + alpha*np.sum(tmp*X1)/150.
    theta_g[2] = theta_g[2] + alpha*np.sum(tmp*X2)/150.
    theta_g[3] = theta_g[3] + alpha*np.sum(tmp*X3)/150.
    J[i] = 0.5*np.sum((Y - dot(X, theta_g))**2) #计算损失函数值    

print("批量梯度下降法：")
print("权重系数：",theta_g)  
print(J.plot(ylim = [0, 50]))