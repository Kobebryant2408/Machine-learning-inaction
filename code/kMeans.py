# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:28:53 2018

@author: 晟玮

特别注意: distEclud函数中sum与np.sum的区别
"""

import numpy as np

#与loadDataSet2相同的作用,但是后者的map函数总是报错
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
    return np.mat(dataArr)
    
#将文本文件转换为列表输出
def loadDataSet2(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

#计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2))) #la.norm(vecA-vecB)

#构建包含k个随机质心的数据集合
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))#create centroid mat
    #构建k个随机质心
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))
    return centroids

#kMeans函数返回质心集合与簇集合
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):#k:创建质心的个数;distMeas:距离计算函数;creatCent:质心创建函数
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))#初始化簇集合,每一行两个元素分别为与该行数据距离最近的质心的索引(mindex)以及该距离的平方
                                            #create mat to assign data points to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)      #初始化k个随机质心
    clusterChanged = True
    #当簇有变化的时候一直计算下去直到簇不再变化
    while clusterChanged:
        clusterChanged = False
        #循环所有数据
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = np.inf;               #初始化最小距离
            minIndex = -1                   #初始化质心的最小索引
            #所有数据分别与随机质心比较计算距离
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; 
                    minIndex = j
            if clusterAssment[i,0] != minIndex: #簇有变化
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print centroids
        #更新质心
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster通过数组过滤获得给定簇(cent)的所有点
            centroids[cent,:] = np.mean(ptsInClust, axis=0) #assign centroid to mean 计算该簇(cent)的均值
    return centroids, clusterAssment