# -*- coding: utf-8 -*-
# hua  
# 2019年3月  
'''Numpy and Linear Algebra'''  

import numpy as np  
import matplotlib.pyplot as plt  
  
#*** 1.矩阵 ***  
A = np.mat([[1,2,3],[4,5,6]]) 
print("\n***1.numpy矩阵***") 
print(A, type(A), A.shape)  
At = A.T  #矩阵转置
print(At)  
print(At.shape)  
  
#*** 2.二维数组 ***
# 其实和矩阵真的挺像的，我暂时还不知道有什么区别  
B = np.array([[1,2,3,4],[5,6,7,8]]) 
print("\n***2.numpy二维数组***") 
print(B,type(B), B.shape)   
print(B+2)  
print(B*2)  
print(B/2)  
print(B-2)   
print(B.T) #转置 

#*** 3.一维数组 ***  
# 就是向量？  
x = np.array([1,0,3])  
y = np.array([1,2,3])  
print("\n***3.numpy一维数组***")  
print(x,y)    
print(x*y) # 点对点地乘   
print(x.dot(y)) # 内积      
print(y.dot(x)) # 内积
diagY = np.diag(y)  
print(diagY) # 常见特殊矩阵：对角阵 
I3 = np.eye(3)  
print(I3)  # 常见特殊矩阵：单位矩阵
  
  
#*** 4.矩阵运算 *** 
A = np.array([[1,2,3,4],[5,6,7,8]]) 
print("\n***4.矩阵运算***")  
print(A)    
B = A  
print(id(A),id(B))    
B = A.copy() # deep copy, not just ref.  
print(id(A),id(B))    
C1 = A*B # 点乘  
print("C1:",C1)  
C2 = A.dot(B.T) # 矩阵乘法  
print("C2:",C2)  
C3 = np.dot(A,B.T) #矩阵乘法，跟C2一样
print("C3:",C3)   
print('Trace of C2 is:', np.trace(C2))  # Trace   
print('Rank of C2 is:',np.linalg.matrix_rank(C2))  # rank, np.rank() 已过时…… 
# ?????  
# 这里会涉及到rank的概念，在线性代数中rank表示秩,请自行查阅相关概念。
  
  
#*** 5.行列式、逆矩阵、伴随矩阵  ***
np.random.seed(0)  
A = np.random.rand(3,3) 
print("\n***5.行列式、逆矩阵、伴随矩阵***") 
print(A)   
print('Det of C2 is:', np.linalg.det(A)) # 行列式 Det  
A_1 = np.linalg.inv(A)  # 逆矩阵 
print("Inverse of A:",A_1)  
print(A.dot(A_1))  
A_star = A_1 * np.linalg.det(A)  # 伴随矩阵  
print("伴随矩阵:",A_star)  
  
  
#*** 6.关于范数 ***
A = np.array([[1,2,3,4],[5,6,7,8]]) 
print("\n***6.关于范数***") 
print(A)  
print(np.linalg.norm(A))  # Default: Frobenius norm  
print(np.sqrt((A**2).sum())) # calc Fro. norm of a matrix by def  
np.random.seed(87)  
x = np.random.rand(3)  
print(x)
print(np.linalg.norm(x))  # 向量的范数  
print(np.sqrt(x.dot(x)))  # 自己 dot 自己，然后开平方  


#*** 7.线性方程组 ***  
A1 = [[1,2,1],[2,-1,3],[3,1,2]]  
A = np.array(A1)
print("\n***7.线性方程组***")   
print(A)   
b = np.array([7,8,18])  # b是一行的话，也能解？？ 
b = b.reshape(-1,1)   # reshpae可以把np.array([7,8,18])转成一列，但transpose没有用
print(type(b),b.shape)  
b = np.array([[7],[8],[18]]) # 列也可以  
print(type(b),b.shape)  
print(b)    
x = np.linalg.inv(A).dot(b)  
print("线性方程组的解x:",x)  
# 另一种方法，直接调用solve  
x2 = np.linalg.solve(A,b)  
print("另一种求解方法x2:",x2)    
# Check  
print(A.dot(x))  
print(A.dot(x2))  
  
  
#*** 8.矩阵的特征值  ***
A = np.array([[1,2,3],[4,5,6],[7,8,9]]) 
print("\n***8.矩阵的特征值***") 
print(A)  
eig_vals,eig_vecs = np.linalg.eig(A)  
print('特征值：',eig_vals)  
print('特征向量：',eig_vecs[:,0],eig_vecs[:,1],eig_vecs[:,2])  
a0 = A.dot(eig_vecs[:,0]) # 矩阵乘以向量  
print(a0 - eig_vals[0]*eig_vecs[:,0])  #输出结果应该是零向量
a1 = A.dot(eig_vecs[:,1]) # 矩阵乘以向量  
print(a1 - eig_vals[1]*eig_vecs[:,1])  #输出结果应该是零向量
a2 = A.dot(eig_vecs[:,2]) # 矩阵乘以向量  
print(a2 - eig_vals[2]*eig_vecs[:,2])  #输出结果应该是零向量
 
  
  
#*** 9.矩阵的奇异值分解 ***   
np.random.seed(10)  
A = np.random.rand(100,100) 
print("\n***9.矩阵SVD***") 
print("Shape of A:",A.shape)
print("Part of A:",A[:2,:2])  
U,S,VH = np.linalg.svd(A)   
print('Part of matrix:',U.dot(np.diag(S)).dot(VH)[:2,:2])  # Part of matrix
plt.plot(S)  
plt.title('SVD of Matrix A')  
plt.grid()  
plt.show()  
  
  
  
