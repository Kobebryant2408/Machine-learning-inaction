# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 09:33:23 2018

@author: 晟玮

重要知识点:standEst函数中的overlap计算方法;
          svdEst函数中的svd方法(构建低维空间)81-83;
          recommed函数中的最后一行中的sorted与lambda;
          
          svd分解选取前几个奇异值可以通过analyse_data函数分析得到(矩阵的能量信息),
          比如svdEst函数中的Sig4中的4,以及imgCompress函数中的numSV的选取.
"""

import numpy as np
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

#下面三个函数计算的相似度都是归一化的
#计算欧式距离    
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))
#计算皮尔逊相关系数
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*np.corrcoef(inA, inB, rowvar = 0)[0][1]
#计算余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

###################################推荐引擎##################################
#对用户未评分物品评分
def standEst(dataMat, user, simMeas, item):#dataMat:数据集;user:用户编号;simMeans:相似度计算方法;item:未评分物品编号
    n = np.shape(dataMat)[1]
    simTotal = 0.0;               #初始化相似度的和
    ratSimTotal = 0.0             #初始化相似度与评分乘积的和
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: 
            continue
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0] #overlap保存未评分物品与已评分物品重合元素(都有评分的元素)的索引
        if len(overLap) == 0: 
            similarity = 0        #该变量表示相似度
        else: 
            similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    else: 
        return ratSimTotal/simTotal

#对用户未评分物品评分(采用svd方法将数据矩阵转换到低维空间)
def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; 
    ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = np.mat(np.eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items ???????????????到底是什么样的低维空间
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: 
            continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    else: return ratSimTotal/simTotal
    
#基于物品的相似度进行推荐    
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst): #estMethod:评分估计的方法
    unratedItems = np.nonzero(dataMat[user,:].A==0)[1]#find unrated items 
    if len(unratedItems) == 0: 
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

###############################图像压缩########################################
#打印矩阵,矩阵中的元素与阈值(thresh)比较????????????为什么打印出来的矩阵是列而不是行
def printMat(inMat, thresh=0.8):   #inMat:待打印矩阵;thresh:阈值
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print ('1'),
            else: 
                print ('0'),
        print ('')

#实现图像压缩        
def imgCompress(numSV=3, thresh=0.8):#numSV:奇异值数目
    myl = []
    for line in open('F:/Anaconda-spyder相关/Ch14/0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print ("****original matrix******")
    printMat(myMat, thresh)
    #利用SVD实现图像压缩
    U,Sigma,VT = la.svd(myMat)
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    #取前numSV个奇异值
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:] #压缩后的数据集
    print ("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)

def analyse_data(Sigma,loopNum=20):
    Sig2 = Sigma**2               #计算奇异值的平方
    SigmaSum = sum(Sig2)          #计算奇异值的平方和
    for i in range(loopNum):
        SigmaI = sum(Sig2[:i+1])  #计算前i个奇异值的平方和
        print('主成分:$S,能量占比:%S%%'%(format(i+1,'2.0f'),format(SigmaI/SigmaSum*100,'4.2f')))