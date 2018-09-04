# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:27:49 2018

@author: 晟玮

重要知识点:pca函数中的eig方法以及构造新的数据集与新的低维空间37-38;
          pca分析中选取几个特征值可以通过analyse_data函数分析得到(方差和累积方差占比).
"""

import numpy as np
import matplotlib.pyplot as plt

#这里用的加载数据集的方法与adaBoost中的方法一致,该章给的map方法总是报错
def loadDataSet(fileName, delim='\t'):
    numFeat = len(open(fileName).readline().split('\t'))
    fr = open(fileName)
    dataArr = []
    for line in fr.readlines():
        lineArr = []
        curline = line.strip().split(delim)
        for i in range(numFeat):
            lineArr.append(float(curline[i]))
        dataArr.append(lineArr)
    #stringArr = [line.strip().split(delim) for line in fr.readlines()] 
    #datArr = [map(float,line) for line in stringArr]    总是出现TypeError: unsupported operand type(s) for /: 'map' and 'int'
    return np.mat(dataArr)

#主成分分析函数
def pca(dataMat, topNfeat=9999999):            #参数topNfeat表示选取的N个特征,即N个主成分
    meanVals = np.mean(dataMat, axis=0)        #计算每一列的均值
    meanRemoved = dataMat - meanVals           #remove mean
    covMat = np.cov(meanRemoved, rowvar=0)     #计算协方差矩阵
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))  #得到协方差矩阵的特征值和特征向量,其中特征值就是方差
    eigValInd = np.argsort(eigVals)                   #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]          #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]               #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects           #transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat               #分别为降维后的数据集,新的数据集空间

def show_picture(dataMat,reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
    plt.show()

#将数据集中的缺失值(NaN)用平均值代替    
def replaceNanWithMean(): 
    datMat = loadDataSet('F:/Anaconda-spyder相关/Ch13/secom.data', ' ')
    numFeat = np.shape(datMat)[1]
    #利用循环对每个特征对应样本中的NaN进行替换(多看一下这个方法!!!!!!!!!)
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal           #set NaN values to mean
    return datMat

def analyse_data(dataMat,topNfeat=20):
    meanVals = np.mean(dataMat,axis=0)
    meanRomoved = dataMat - meanVals
    covMat = np.cov(meanRomoved,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    
    #topNfeat = 20
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    cov_all_score = float(sum(eigVals)) #计算总方差(特征值的和)
    sum_cov_score = 0                   #初始化累积方差
    #从最大特征值(方差)循环计算方差占比和累积方差占比
    for i in range(0,len(eigValInd)):
        line_cov_score = float(eigVals[eigValInd[i]])
        sum_cov_score += line_cov_score
        print('主成分:%s,方差占比:%s%%,累积方差占比:%s%%'%(format(i+1,'2.0f'),format(line_cov_score/cov_all_score*100,'4.2f'),format(sum_cov_score/cov_all_score*100,'4.2f')))
        