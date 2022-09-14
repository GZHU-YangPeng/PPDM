import numpy as np
import pandas as pd
import binascii  #二进制与AScall码转换

from sklearn.datasets import make_blobs       #数据
from sklearn.datasets import make_moons
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt                #画图工具

from sklearn.cluster import KMeans             #KMeans聚类
from sklearn.cluster import DBSCAN  
from sklearn.cluster import AgglomerativeClustering    #凝聚聚类

import math
from sklearn import metrics                   #包含各种评价指标函数，比如ARI,AMI,FMI等

# import torch
import copy

#获取数据
def get_data(kind,noisy):
    
    if (kind=="blobs"):
        #随机在数据集中取数据
        d, l = make_blobs(n_features=2, n_samples=300, centers=3, random_state=2, cluster_std=[1.2, 0.6, 1.2])
      #  plt.scatter(d[:,0],d[:,1],c=l)    #画出原始数据，及真实标签(用不同颜色代替)
        return d,l
    
    elif (kind=="moons"):
        d,l = make_moons(n_samples=200,noise=noisy, random_state=0)
       # plt.scatter(d[:,0],d[:,1],c=l)
        return d,l
    
    elif (kind=="gaussian"):
        d,l = mmake_gaussian_quantiles(n_samples=200)
      #  plt.scatter(d[:,0],d[:,1],c=l)
        return d,l
    
    elif (kind=="iris"):
        iris=load_iris()
        d = iris.data
        l = iris.target
       # plt.scatter(d[:,1],d[:,2],c=l)
        return d,l
    
    elif (kind=="breast_cancer"):
        breast_cancer = load_breast_cancer()
        d = breast_cancer.data
        l = breast_cancer.target
       # plt.scatter(d[:,1],d[:,2],c=l)
        return d,l    
    
    elif (kind=="digits"):
        digits = load_digits()
        d = digits.data
        l = digits.target
      #  plt.scatter(d[:,1],d[:,2],c=l)
        return d,l  
    
    
#聚类方法
def kmeans(d,class_number):
    kmeans = KMeans(n_clusters=class_number)     #设定 k 值，类别数
    kmeans.fit(d)                     #训练
    km_pred = kmeans.predict(d)         #预测标签
    return km_pred

def Agglomerative(d,class_number):
    agg = AgglomerativeClustering(n_clusters = class_number)
    ag_pred = agg.fit_predict(d)
    return ag_pred

def Dbscan(d,eps_num,min_samples_num):
    print("注意设置参数eps,min_samples")           #eps=0.3, min_samples=10
    db = DBSCAN(eps=eps_num, min_samples=min_samples_num).fit(d)  #eps表示数据之间的连通距离,设太大则只有1类
    labels = db.labels_                          #min_samples
    return labels





# #算出聚类指标所需的a，b，c，d，入
# #利用tensor计算，要求2个输入必须为float, int, uint8, bool等数组，不支持字符数组
# def compute_tensor(km_pred,ag_pred):                  #计算a,b,c,d,lamda
#     label_A = torch.tensor(km_pred)
#     label_B = torch.tensor(ag_pred )
    
#     #sam_A表示在A的分区下，两两数据 同类为1，不同类为0
#     sam_A = (label_A.unsqueeze(0) == label_A.unsqueeze(1)).int()  
#     sam_B = (label_B.unsqueeze(0) == label_B.unsqueeze(1)).int()

#     #unsam_A表示在聚类A的预测标签下，两两数据 不同类为1，同类为0
#     unsam_A = (label_A.unsqueeze(0) != label_A.unsqueeze(1)).int()  
#     unsam_B = (label_B.unsqueeze(0) != label_B.unsqueeze(1)).int()
    
#     s_a = (sam_A * sam_B)               # 矩阵点乘，对应位置相乘 表示xi,xj在聚类A中同类，在聚类B也同类
#     a = (s_a.sum() - s_a.trace() )/2        #去掉对角线的1， 再求上三角部分
    
#     s_b = (sam_A * unsam_B)             # 矩阵点乘，对应位置相乘 表示xi,xj在聚类A中同类，在聚类不同类
#     b = (s_b.sum() - s_b.trace() )/2        #去掉对角线的1， 再求上三角部分
    
#     s_c = (unsam_A * sam_B)             # 矩阵点乘，对应位置相乘 表示xi,xj在聚类A中同类，在聚类不同类
#     c = (s_c.sum() - s_c.trace() )/2        #去掉对角线的 1， 再求上三角部分
    
#     s_d = (unsam_A * unsam_B)             # 矩阵点乘，对应位置相乘 表示xi,xj在聚类A中不同类，在聚类不同类
#     d = (s_d.sum() - s_d.trace() )/2        #去掉对角线的1， 再求上三角部分
    
#     a,b,c,d = a.float(),b.float(),c.float(),d.float()

#     lamda = a+b+c+d
    
#     return a,b,c,d,lamda
    

#算出聚类指标所需的a，b，c，d，入
#利用numpy计算，2个输入支持float, int, uint8, bool等，支持字符数组
    
def compute_numpy(km_pred,ag_pred):                  #计算a,b,c,d,lamda  

    pred_A = copy.deepcopy(km_pred)
    pred_B = copy.deepcopy(ag_pred)
    
    label_A_row = km_pred.reshape((1,len(pred_A)))   #利用reshape改变数组形状，变成(1,n)
    label_A_col = km_pred.reshape((len(pred_A),1))   #利用reshape改变数组形状，变成(n,1)
    
    label_B_row = ag_pred.reshape((1,len(pred_B)))
    label_B_col = ag_pred.reshape((len(pred_B),1))
    
#sim_A表示在A的聚类分区下，两两数据 同类为1，不同类为0
    sim_A = (label_A_row == label_A_col)    #类似tensor中(A.unsqueeze(0)== A.unsqueeze(1)
    sim_B = (label_B_row == label_B_col)   
    #由 bool数组 -> int数组
    sim_A = sim_A.astype(np.int)
    sim_B = sim_B.astype(np.int)

    row,col = np.diag_indices_from(sim_A)   #将对角线置零
    sim_A[row,col] = 0
    row,col = np.diag_indices_from(sim_B)   #将对角线置零
    sim_B[row,col] = 0
    
### unsim_A表示在聚类A的预测标签下，两两数据 不同类为1，同类为0
    unsim_A = 1-sim_A   #若相似为1，则不相似为0
    unsim_B = 1-sim_B
    
    row,col = np.diag_indices_from(unsim_A)   #将对角线置零
    unsim_A[row,col] = 0
    row,col = np.diag_indices_from(unsim_B)   #将对角线置零
    unsim_B[row,col] = 0 

    a = ((sim_A * sim_B).sum()  ) /2           # 矩阵点乘，对应位置相乘 表示xi,xj在聚类A中同类，在聚类B也同类        /2 求上三角部分
    
    b = ((sim_A * unsim_B).sum()  ) /2           # 矩阵点乘，对应位置相乘 表示xi,xj在聚类A中同类，在聚类不同类    求上三角部分
    
    c = ((unsim_A * sim_B).sum()  ) /2           # 矩阵点乘，对应位置相乘 表示xi,xj在聚类A中同类，在聚类不同类    求上三角部分
    
    d = ((unsim_A * unsim_B).sum() )/2            # 矩阵点乘，对应位置相乘 表示xi,xj在聚类A中不同类，在聚类不同类  求上三角部分
    
    a,b,c,d = a.astype("float"), b.astype("float"), c.astype("float"), d.astype("float")
    
    lamda = a+b+c+d
#     print(a,b,c,d,lamda)
    return a,b,c,d,lamda


##################变换类型
def array2bytes(some_arr):    #np.array -> bytes
    
    str1 = str(some_arr)    #将array数组 -> 字符串   但会保留里面的换行符和 "[]"
        #如：'[1 1 1 0 1 0 1 0 0 1 \n 1 0]' 
    str2 = str1.replace(' ', '')            # 去除字符串中无用的字符
    str3 = str2.replace('\n', '') 
    str4 = str3.replace('[', '') 
    str5 = str4.replace(']', '') 
    byte1 = bytes(str5, encoding = "utf8")  # str -> bytes(b'11101')
    
    return byte1

def array2str(some_arr):    #np.array -> bytes
    
    str1 = str(some_arr)    #将array数组 -> 字符串   但会保留里面的换行符和 "[]"
        #如：'[1 1 1 0 1 0 1 0 0 1 \n 1 0]' 
    str2 = str1.replace(' ', '')            # 去除字符串中无用的字符
    str3 = str2.replace('\n', '') 
    str4 = str3.replace('[', '') 
    str5 = str4.replace(']', '') 
    return str5
    
def bytes2str(some_bytes):
    some_str = some_bytes.decode('utf8')        #bytes -> str
    return some_str


def str2array(some_str):
    some_byte = bytes(some_str, encoding = "utf8")    # str -> bytes

    some_array = np.frombuffer(some_byte, dtype=np.uint8)      #bytes -> array数组

    return some_array


def bytes2array(some_bytes):

    some_array = np.frombuffer(some_bytes, dtype=np.uint8)      #bytes -> array数组

    return some_array


def str2char_array(some_str):
    some_list=[]
    
    for j in some_str:
        some_list.append(j)
        
    some_arr = np.array(some_list)
    
    return some_arr

def hex2decimal(hex_str):                  #16进制字符串 -> str
    
    temp = []
    for i in hex_str:
        temp.append(int(i, 16))   #int(i, 16)将每个16进制字符转为10进制

    decimal_arr = np.array(temp)   
    return decimal_arr


    




    
#聚类指标函数    
def ARI(km_pred,ag_pred):
    
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
#     a,b,c,d,lamda = compute_tensor(km_pred,ag_pred)
    
    num = ( a-(a+b)*(a+c)/lamda) / ( (a+b+a+c)/2 - (a+b)*(a+c)/lamda ) 
    return (num)

def B(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    
    num =( lamda**2 - lamda*(b+c)+(b-c)**2) / lamda**2
    return (num)

def CZ(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    
    num =2*a / (2*a+b+c)
    return (num)

def FM(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    
    num = a / math.sqrt( (a+b)*(a+c) )
    return (num)

def G(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    
    num = ( lamda * a - (a+b)*(a+c) ) / math.sqrt((a+b)*(a+c)*(c+d)*(b+d)) 
    return (num)

def GK(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = ( a*d - b*c ) / ( a*d + b*c )
    return (num)

def GL(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = ( a+d ) / (a + 0.5*(b+c) +d ) 
    return (num)

def H(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = (( a+d) - (b+c) ) / lamda
    return (num)

def J(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = a / (a+b+c)
    return (num)

def K(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = ( a/(a+b) + a/(a+c) ) / 2
    return (num)

def MC(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = (a**2 - b*c) / ((a+b)*(a+c))
    return (num)

def P(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = (a*d - b*c) / ((a+b)*(a+c)*(c+d)*(b+d))
    return (num)

def PE(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = (a*d - b*c) / ((a+c)*(b+d))
    return (num)

def R(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = (a+d) / (lamda)
    return (num)

def RR(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = (a) / (lamda)
    return (num)

def RT(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = (a+d) / (a + 2*(b+c) +d)
    return (num)

def SS1(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num=( (a/(a+b)) + (a/(a+c)) + (d/(d+b)) + (d/(d+c)) ) /4
    return (num)

def SS2(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = a / (a + 2*(b+c))
    return (num)

def SS3(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = (a*d ) /( math.sqrt( (a+b)*(a+c)*(d+b)*(d+c) ) )
    return (num)

def W1(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = (a) /( (a+b) )
    return (num)

def W2(km_pred,ag_pred):
    a,b,c,d,lamda = compute_numpy(km_pred,ag_pred)
    num = (a) /( (a+c) )
    return (num)




# 计算21个评价指标下的 sim(A,B)与sim(A',B')
def compute_sim(pre_k, pre_n, post_k, post_n):
    
#     print("ARI(A,B):", ARI(pre_k, pre_n),"     ARI(A',B'):",ARI(post_k, post_n) )  #(-1,1]
#     print("B(A,B)  :", B(pre_k, pre_n),"     B(A',B')  :",B(post_k, post_n) )
#     print("CZ(A,B) :", CZ(pre_k, pre_n),"     CZ(A',B') :",CZ(post_k, post_n) )
#     print("FM(A,B) :", FM(pre_k, pre_n),"     FM(A',B') :",FM(post_k, post_n) )
#     print("G(A,B)  :", G(pre_k, pre_n),"     G(A',B')  :",G(post_k, post_n) )      #[-1,1]
#     print("GK(A,B) :", GK(pre_k, pre_n),"     GK(A',B') :",GK(post_k, post_n) )    #[-1,1]
#     print("GL(A,B) :", GL(pre_k, pre_n),"     GL(A',B') :",GL(post_k, post_n) )
#     print("H(A,B)  :", H(pre_k, pre_n),"     H(A',B')  :",H(post_k, post_n) )      #[-1,1]
#     print("J(A,B)  :", J(pre_k, pre_n),"     J(A',B')  :",J(post_k, post_n) )
#     print("K(A,B)  :", K(pre_k, pre_n),"     K(A',B')  :",K(post_k, post_n) )
#     print("MC(A,B) :", MC(pre_k, pre_n),"     MC(A',B') :",MC(post_k, post_n) )    #[-1,1]
#     print("P(A,B)  :", P(pre_k, pre_n)," P(A',B')  :",P(post_k, post_n) )          #[-1,1]
#     print("PE(A,B) :", PE(pre_k, pre_n),"     PE(A',B') :",PE(post_k, post_n) )    #[-1,1]
#     print("R(A,B)  :", R(pre_k, pre_n),  "     R(A',B')  :",R(post_k, post_n) )
#     print("RR(A,B) :", RR(pre_k, pre_n), "     RR(A',B') :",RR(post_k, post_n) )
#     print("RT(A,B) :", RT(pre_k, pre_n), "     RT(A',B') :",RT(post_k, post_n) )
#     print("SS1(A,B):", SS1(pre_k, pre_n),"     SS1(A',B'):",SS1(post_k, post_n) )
#     print("SS2(A,B):", SS2(pre_k, pre_n),"     SS2(A',B'):",SS2(post_k, post_n) )
#     print("SS3(A,B):", SS3(pre_k, pre_n),"     SS3(A',B'):",SS3(post_k, post_n) )
#     print("W1(A,B) :", W1(pre_k, pre_n), "     W1(A',B') :",W1(post_k, post_n) )
#     print("W2(A,B) :", W2(pre_k, pre_n), "     W2(A',B') :",W2(post_k, post_n) )
    
    list_pre = np.array([ [ARI(pre_k, pre_n)],[B(pre_k, pre_n)],[CZ(pre_k, pre_n)],[FM(pre_k, pre_n)],
                         [G(pre_k, pre_n)],[GK(pre_k, pre_n)],[GL(pre_k, pre_n)],[H(pre_k, pre_n)],
                         [J(pre_k, pre_n)],[K(pre_k, pre_n)],[MC(pre_k, pre_n)],    # [P(pre_k, pre_n)],
                         [PE(pre_k, pre_n)],[R(pre_k, pre_n)],                      #[RR(pre_k, pre_n)],
                         [RT(pre_k, pre_n)],[SS1(pre_k, pre_n)],[SS2(pre_k, pre_n)],[SS3(pre_k, pre_n)],
                         [W1(pre_k, pre_n)],[W2(pre_k, pre_n)]],dtype = 'float32')
    
    
    list_post = np.array([ [ARI(post_k,post_n)],[B(post_k,post_n)],[CZ(post_k,post_n)],[FM(post_k,post_n)],
                           [G(post_k,post_n)],[GK(post_k,post_n)],[GL(post_k, post_n)],[H(post_k, post_n)],
                           [J(post_k,post_n)],[K(post_k, post_n)],[MC(post_k, post_n)],  #[P(post_k, post_n)],
                           [PE(post_k,post_n)],[R(post_k, post_n)],                      #[RR(post_k,post_n)],
                           [RT(post_k,post_n)],[SS1(post_k,post_n)],[SS2(post_k, post_n)],[SS3(post_k, post_n)],
                           [W1(post_k, post_n)],[W2(post_k, post_n)]],dtype = 'float32')
    
    return list_pre, list_post