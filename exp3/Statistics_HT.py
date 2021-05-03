# -*- coding: utf-8 -*-

'''
1.描述统计学
  –集中趋势
  –离散趋势
  –偏态
2.假设检验
  –基本原理
  –p value
统计学可以分为：描述统计学与推断统计学
描述统计学：使用特定的数字或图表来体现数据的集中程度和离散程度。例：每次考试算的平均分，最高分，各个分段的人数分布等，也是属于描述统计学的范围。
推断统计学：根据样本数据推断总体数据特征。例：产品质量检查，一般采用抽检，根据所抽样本的质量合格率作为总体的质量合格率的一个估计。
应用：统计学的应用十分广泛，可以说，只要有数据，就有统计学的用武之地。目前比较热门的应用：经济学，医学，心理学等。
'''

# -*- coding: utf-8 -*-
from pandas import Series, DataFrame
import numpy as np
import math
import matplotlib.pyplot as plt

#===========1. 描述统计学==================================
a=[98,83,65,72,79,76,75,94,91,77,63,83,89,69,64,78,63,86,91,72,71,72,70,80,65,70,62,74,71,76]
print(np.mean(a)) # 75.9666666667
print(np.sort(a))  
# [62 63 63 64 65 65 69 70 70 71 71 72 72 72 74 75 76 76 77 78 79 80 83 83 86 89 91 91 94 98]
print( np.sort(a)[14:16] )  # 包含起始位置、不包括结束位置  [74 75]
print( np.mean(np.sort(a)[14:16]) )  # 74.5

#众数——数据中出现次数最多的数（所占比例最大的数）
def get_mode(arr):
    mode = [];  
    arr_appear = dict((a, arr.count(a)) for a in arr);  # 统计各个元素出现的次数  
    if max(arr_appear.values()) == 1:
        return; 
    else:
        for k, v in arr_appear.items():  # 出现次数最多的数，就是众数  
            if v == max(arr_appear.values()):  
                mode.append(k);
    return mode;

print(get_mode(a));  # [72]
print(np.var(a));  # 方差 93.2988888889
print(np.std(a));  # 计算样本标准差 9.65913499693

a=Series(a)
print(a.skew())  # 计算偏度  0.574289187335
print(a.kurt())   # 计算峰度 -0.428723747954

'''
偏度和峰度都是统计量 
一阶矩是随机变量的期望，二阶矩是随机变量平方的期望
偏度Skewness(三阶) —— 三阶中心距除以标准差的三次方
峰度Kurtosis (四阶) ——概率密度在均值处峰值高低的特征，常定义四阶中心矩除以方差的平方，减去三；
'''

print(a.describe()) # 描述
'''
count    30.000000
mean     75.966667
std       9.824260
min      62.000000
25%      70.000000
50%      74.500000
75%      82.250000
max      98.000000
dtype: float64
'''

df = DataFrame({'data1' : np.random.randn(5),
                'data2' : np.random.randn(5)})
print(df.cov()) # E[(X-E(X))(Y-E(Y))]称为随机变量X和Y的协方差，记作COV(X，Y)，即COV(X，Y)=E[(X-E(X))(Y-E(Y))^T]
print(df.corr())

#计算person相关系数
a = np.array([[1, 1, 2, 2, 3],  
       [2, 2, 3, 3, 5],  
       [1, 4, 2, 2, 3]])  
# 可计算行与行之间的相关系数，np.corrcoef(a,rowvar=0)用于计算各列之间的相关系数，输出为相关系数矩阵。
print( np.corrcoef(a) ) 
'''
[[ 1.      0.9759  0.1048]
 [ 0.9759  1.      0.179 ]
 [ 0.1048  0.179   1.    ]]
'''
print( np.corrcoef(a,rowvar=0) )  
'''
[[ 1.    -0.189  1.     1.     1.   ]
 [-0.189  1.    -0.189 -0.189 -0.189]
 [ 1.    -0.189  1.     1.     1.   ]
 [ 1.    -0.189  1.     1.     1.   ]
 [ 1.    -0.189  1.     1.     1.   ]]
'''
print( ' ######END 描述统计学######### ' )


#=================2. 假设检验 ==============================
'''
基本思想
    –小概率思想
    –反证法思想
零假设与备择假设——无罪推定原理
两类错误
    –第一类错误
    –第二类错误

假设检验的基本步骤
    1. 提出零假设
    2. 建立检验统计量
    3. 确定否定域/计算p-value
    4. 得出结论

'''
from scipy import stats as ss
sa=[10.1,10,9.8,10.5,9.7,10.1,9.9,10.2,10.3,9.9]
t1=ss.ttest_1samp(sa, 10) #单一样本T检验，H0：均值为10
print(t1)

rvs1 = ss.norm.rvs(loc=5,scale=10,size=500)
rvs2 = ss.norm.rvs(loc=5,scale=10,size=500)
t2=ss.ttest_ind(rvs1,rvs2) #两独立样本t检验-ttest_ind
print(t2)

rvs3 = ss.norm.rvs(loc=5, scale=20, size=500)
t3=ss.ttest_ind(rvs1, rvs3, equal_var = False) #方差不一样
print(t3)


rvs1 = ss.norm.rvs(loc=5,scale=10,size=500)
rvs2 = ss.norm.rvs(loc=5,scale=10,size=500) + ss.norm.rvs(scale=0.2,size=500)
t4=ss.ttest_rel(rvs1,rvs2) #配对样本T检验-ttest_rel
print(t4)

