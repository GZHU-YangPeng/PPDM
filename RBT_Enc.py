import numpy as np
import pandas as pd
import math
from cluster import*
# from sklearn.preprocessing import scale    #导入用于数据标准化的模块

# from sklearn.preprocessing import MinMaxScaler     # 最大最小标准化# 标准化
from sklearn.preprocessing import scale

def pair(D):         #接收数据矩阵
    i = 0
    j = 0
    r,c = D.shape    # c表示总列数
    if(c % 2 ==0):   #偶数列       (5,2)
        v = np.random.randn(int(c/2), 2, r)   #用来存储配好的列对，
    else:                #奇数列      12 34 5_  循环3次
        v = np.random.randn(int((c+1)/2), 2, r)  #用来存储配好的对，每个元素为 (2,行数)

    # 偶数列，如 6，则最后一个i=4(下一个i+2为6，不满足循环条件)，刚好最后两行配对
    # 奇数列，如 7，则最后一个i=6(下一个i+2为8，不满足循环条件)，最后一行与第一行配对     
        
        
    if(c%2==0):    # 偶数列, 有c/2对特征
        for n in range(int(c/2)):    # 可以与相邻的列配对，也可以别的方式
            v[j] = (np.r_[ D[:, i], D[:, (i+1)%c]]).reshape(2,r) #把(150,2)转为(2,150)
            
            j += 1
            i += 2   #自己设定，这里为相邻  12 34 56

    else:    #奇数
        for n in range(int(c/2)+1): 
            v[j] = (np.r_[ D[:,i], D[:, (i+1)%c]]).reshape(2,r) #把(150,2)转为(2,150)
            j += 1
            i += 2
            
    return v         #得到v之后，后面需要新的数组形状可按v设置

from sympy.abc import x  
from sympy import*
import sympy
from sympy import symbols,solve
import copy 

# x = symbols("x")
# R = np.array([[sympy.cos(x),sympy.sin(x)],[-sympy.sin(x),sympy.cos(x)]])
# R = np.array([[np.cos(x),np.sin(x)],[-np.sin(x),np.cos(x)]])

def rotate(v):              #R为旋转矩阵，v存的是多个成对数据
    x = symbols("x")
    R = np.array([[sympy.cos(x),sympy.sin(x)],[-sympy.sin(x),sympy.cos(x)]])
    r,c,z = v.shape
    ### 第一步：将每对数据对旋转存入v_post
#     v_post = copy.deepcopy(v)       
    v_post=[]
    for i in range( len(v) ):      #v的一维数目就是数组对数目，也就是需要的角度数
        v_post.append(R @ v[i])    #(2,2) @ (2,x) 用旋转矩阵对每对数据对旋转，存入v_post 
    
    v_post = np.array(v_post)
    return v_post


#方差
def var(b):
    n = len(b)
    b_sum = 0
    for i in range(n):
        b_sum += b[i]
    b_ave = b_sum/n
    s = 0
    for i in range(n):
        s+=(b[i] - b_ave)**2
    s = s/(n-1)
    return s

import random

def PST(D_pair, D_rotate):   # 选出每对特征旋转的角度
    theta = []
    s=0  
    for p in range(len(D_pair)):
        # 分别为两个特征的方差，带未知数角度x
        f1 = var(D_pair[p][0] - D_rotate[p][0])  #var(age - age')
        f2 = var(D_pair[p][1] - D_rotate[p][1]) #var(heart_ - heart_')
        
        # 最大方差为旋转 180度，相减就是2倍
        var1_max = var(2*D_pair[p][0])  
        var2_max = var(2*D_pair[p][1])
        
        # 随机选取一个方差阈值（方差阈值代表加密强度）
        pst1 = random.uniform(var1_max/3, var1_max)
        pst2 = random.uniform(var2_max/3, var2_max)
        
        for i in range(500):
            s+=1
            #注意，数据旋转是顺时针的，需要换算为顺时针的角度
            a = random.uniform(0,360)/180*np.pi   #换算为弧度
#             a = random.uniform(-np.pi, np.pi)

            # 选取满足两个特征方差阈值的角度
            #用evalf函数，传入变量的值，对表达式进行求值
            if (f1.evalf(subs={x:a}) >=pst1) and (f2.evalf(subs={x:a}) >=pst2):
            
    #             print(f1.evalf(subs={x:a}), f2.evalf(subs={x:a}))

                theta.append(a)
                break
            elif(s==100):
                print("No angle found to match")
                
    return theta


def Ciphertext_D(D_pair, theta, num_features):
    ## num_features为原数据特征数目
    for i in range(len(D_pair)):
        
        R = np.array([[np.cos(theta[i]),np.sin(theta[i])],[-np.sin(theta[i]),np.cos(theta[i])]])
        D_pair[i] = R @ D_pair[i]  # 成对旋转
        
    a,b,c = D_pair.shape
    
    # 此时已得到旋转后的特征对, 将其前两个维度合并，此时每个特征按行排列,再转置即可
    D_cipher = (D_pair.reshape(a*b, c) ).T
    
    # 若原数据特征为奇数，则需删除一列
    if(num_features % 2==1):
        return (D_cipher[:, :-1]) 
    
    else:
        
        return D_cipher
    

def RBTD(D):
    D_std = scale(D, axis=0)            
    D_pair = pair(D_std)        
    D_rotate = rotate(D_pair)   
    theta = PST(D_pair, D_rotate)   

    D_cipher = Ciphertext_D(D_pair, theta, D.shape[1])
    
    return D_cipher