# -*- coding: utf-8 -*-

'''假设需要求解目标函数是 func(x) = x * x 的极小值，由于 func 是一个凸函数，
因此它唯一的极小值同时也是它的最小值，其一阶导函数 为 dfunc(x) = 2 * x。'''

import numpy as np
import matplotlib.pyplot as plt

#目标函数:y=x^2
def func(x):
    return x**2

#目标函数一阶导数:dy/dx=2*x
def dfunc(x):
    return 2*x

def GD(x_start, df, epochs, lr):
    """
    梯度下降法。给定起始点与目标函数的一阶导函数，求在epochs次迭代中x的更新值
    :x_start: x的起始点
    :df: 目标函数的一阶导函数
    :epochs: 迭代周期
    :lr: 学习率
    :return: x在每次迭代后的位置（包括起始点），长度为epochs+1
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    for i in range(epochs):       
        dx = - lr*df(x) # dx表示x要改变的幅度
        x += dx
        xs[i+1] = x
    return xs

def GD_demo0():
    x_start = -5
    epochs = 10
    lr = 0.3
    x = GD(x_start, dfunc, epochs, lr=lr)
    print("x_start:",x_start, "learning rate:",lr, "epochs:",epochs)
    print(x)  
GD_demo0()

def GD_demo1():
    line_x = np.linspace(-5, 5, 100)
    line_y = func(line_x)
    x_start = -5
    epochs = 15
    lr = 0.2 #learning rate
    x = GD(x_start, dfunc, epochs, lr=lr)

    plt.figure(figsize=(6,6))

    plt.plot(line_x, line_y, c='b')
    plt.plot(x, func(x), c='r', label='lr=%.2f'%(lr))
    plt.scatter(x, func(x), c='r', )
    plt.legend()
    plt.show()  
GD_demo1()
