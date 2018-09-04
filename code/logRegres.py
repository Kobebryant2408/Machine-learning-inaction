# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:12:03 2017

@author: 晟玮

主要内容：从疝气病症预测病马的死亡率
特别注意：列表，数组与矩阵的区别
>>>a1=[1,2,3]   #列表
>>>a2=array(a1)
>>> a2
array([1, 2, 3]) #数组
>>>a3=mat(a1)
>>> a3
matrix([[1, 2, 3]]) #矩阵
>>> a4=a2.tolist() #数组转列表（一维列表）
>>> a4
[1, 2, 3]
>>> a5=a3.tolist() #矩阵转列表（二维列表）
>>> a5
[[1, 2, 3]]
"""

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('F:/Anaconda-spyder相关/Ch05/testSet.txt')
    
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
        
    return dataMat, labelMat#输出的列表类型

def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

#梯度上升算法求最佳参数
def gradAscent(dataMatIn, classesLabels):
    dataMatrix = np.mat(dataMatIn)#列表转换为矩阵
    labelMat = np.mat(classesLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1)) #创建数组

#利用矩阵运算计算代价函数J(w)的最小值    
    for i in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    
    return weights

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    
    x = np.arange(-3.0, 3.0, 0.1)
    weights = weights.getA()#将矩阵转换为数组的函数
    y = (-weights[0][0]-weights[1][0]*x)/weights[2][0]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#随机梯度上升算法   
def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    dataMatrix = np.array(dataMatrix)#列表转换为数组
    alpha = 0.01
    weights = np.ones(n) #创建数组
    
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))#数组与数组对应位置相乘
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    
    weights = np.mat(weights).transpose()
    return weights

#改进随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter = 500):
    m,n = np.shape(dataMatrix)
    dataMatrix = np.array(dataMatrix)
    weights = np.ones(n)
    
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    
    weights = np.mat(weights).transpose()
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

#训练集训练出最优权值，测试集验证错误率   
def colicTest():
    frTrain = open('F:/Anaconda-spyder相关/Ch05/horseColicTraining.txt')
    frTest = open('F:/Anaconda-spyder相关/Ch05/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')#对字符串的操作,输出的列表中元素仍是字符串
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(lineArr, trainWeights))!=int(currLine[21]):
            errorCount += 1
    
    errorRate = (float(errorCount)/numTestVec)
    print('the error rate is:',errorRate)
    return errorRate#, trainWeights

#重复验证十次 计算平均错误率    
def muitiTest():
    numTests = 10
    errorSum = 0.0
    
    for i in range(numTests):
        errorSum += colicTest()
    
    print('after %d iterations the average error rate is: %f' % (numTests, errorSum/float(numTests)))
    