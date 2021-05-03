# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


#1. 样本均值计算
#计算平均值
x = np.arange(1, 11)
print(x)  # [ 1  2  3  4  5  6  7  8  9 10]
mean = np.mean(x)
print(mean)  # 5.5

#对空值的处理，nan stands for 'Not-A-Number'
x_with_nan = np.hstack((x, np.nan))
print(x_with_nan)  # [  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  nan]
mean2 = np.mean(x_with_nan)
print(mean2)  # nan，直接计算没有结果
mean3 = np.nanmean(x_with_nan)
print(mean3)  # 5.5

#计算几何平均值
x2 = np.arange(1, 11)
print(x2)  # [ 1  2  3  4  5  6  7  8  9 10]
geometric_mean = stats.gmean(x2)
print(geometric_mean)  # 4.52872868812，几何平均值小于等于算数平均值

#2. 样本方差

#参考：https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html
#参考：https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html

data = np.arange(7, 14)
print(data)  # [ 7  8  9 10 11 12 13]

## 计算方差
# 直接使用样本二阶中心距计算方差，分母为n
var_n = np.var(data)  # 默认，ddof=0
print(var_n) # 4.0
# 使用总体方差的无偏估计计算方差，分母为n-1
var_n_1 = np.var(data, ddof=1)  # 使用ddof设置自由度的偏移量
print(var_n_1) # 4.67

## 计算标准差
std_n = np.std(data, ddof=0)
std_n_minus_1 = np.std(data, ddof=1)  # 使用ddof设置自由度的偏移量
print(std_n, std_n_minus_1)  # 2.0, 2.16
print(std_n**2, std_n_minus_1**2)  # 4.0, 4.67

#3. 一个测试
#测试的是样本均值的标准差随着样本量的变化而发生的变化，如果方差缩小n倍，那么理论上标准差会缩小sqrt(n)倍
import numpy as np
from scipy import stats

def mean_and_std_of_sample_mean(ss=[], group_n=100):
    """
    不同大小样本均值的均值以及标准差
    """
    norm_dis = stats.norm(0, 2)  # 定义一个均值为0，标准差为2的正态分布
    for n in ss:
        sample_mean = []  # 收集每次取样的样本均值
        for i in range(group_n):
            sample = norm_dis.rvs(n)  # 取样本量为n的样本
            sample_mean.append(np.mean(sample))  # 计算该组样本的均值
        print(np.std(sample_mean), np.mean(sample_mean))

sample_size = [1, 4, 9, 16, 100]  # 每组试验的样本量
group_num = 10000
mean_and_std_of_sample_mean(ss=sample_size, group_n=group_num)