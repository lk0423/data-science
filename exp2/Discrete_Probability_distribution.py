# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#1. 二项分布
def binom_dis(n=1, p=0.1):
    """
    二项分布，模拟抛硬币试验
    :param n: 实验总次数
    :param p: 单次实验成功的概率
    :return: 试验成功的次数
    """
    bin_dis = stats.binom(n, p) #产生一个二项分布
    X = bin_dis.rvs()  # 符合该分布的一个随机变量X,取值表示出现正面的次数
    print(X)  #每次结果会不一样
    prob_10 = bin_dis.pmf(10) #pmf: probability mass function，概率分布列，表示正面出现10次的概率
    print(prob_10)  # 0.117
    XV = bin_dis.rvs(size=5) #5维随机向量，每个随机向量都服从分布bin_dis
    print(XV)

    

binom_dis(n=20, p=0.6)

#2. 泊松分布
def poisson_pmf(mu=3):
    """
    泊松分布
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html#scipy.stats.poisson
    :param mu: 单位时间（或单位面积）内随机事件的平均发生率
    :return:
    """
    poisson_dis = stats.poisson(mu)
    x = np.arange(poisson_dis.ppf(0.001), poisson_dis.ppf(0.999))
    print(x)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, poisson_dis.pmf(x), 'bo', ms=8, label='poisson pmf')
    ax.vlines(x, 0, poisson_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('PMF of poisson distribution(mu={})'.format(mu))
    plt.show()

poisson_pmf(mu=8)

#3. 自定义分布函数
#定义一个取值范围为{0,1,2,3,4,5,6}的离散型随机分布以及该分布的PMF图
def custom_made_discrete_dis_pmf():
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html
    :return:
    """
    xk = np.arange(7)  # 所有可能的取值
    print(xk)  # [0 1 2 3 4 5 6]
    pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)  # 各个取值的概率
    custm = stats.rv_discrete(name='custm', values=(xk, pk))

    fig, ax = plt.subplots(1, 1)
    ax.plot(xk, custm.pmf(xk), 'ro', ms=8, mec='r')
    ax.vlines(xk, 0, custm.pmf(xk), colors='r', linestyles='-', lw=2)
    plt.title('Custom discrete distribution(PMF)')
    plt.ylabel('Probability')
    plt.show()

custom_made_discrete_dis_pmf()