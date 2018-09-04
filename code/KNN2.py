# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:09:57 2017

@author: 晟玮

主要内容：约会网站预测和手写识别系统
"""

import numpy as np
import operator
from os import listdir

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#本程序最关键的KNN分类器程序
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
   
    sortedDistIndicies = distances.argsort()
    
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    
    return sortedClassCount[0][0] 
        
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, : ] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    
    return returnMat, classLabelVector

#数据归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)#参数0表示得到每列的最小值
    maxVals = dataSet.max(0)
    ranges= maxVals - minVals
    
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals

#利用数据集对算法进行测试
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('F:\Anaconda-spyder相关\DatingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, : ], normMat[numTestVecs:m, : ], datingLabels[numTestVecs:m], 10)
        print("the classfier came back with: %d, the real answer is %d" %(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("the total error rate is : %f" %(errorCount/float(numTestVecs)))

#利用模型进行预测    
def clasdifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = eval(input("percentage of time spent playing video game?"))
    ffMiles = eval(input("frequent filer miles earned per year?"))
    iceCream = eval(input("liters of ice creams consumed per year?"))
    
    datingDataMat, datingLabels = file2matrix('F:\Anaconda-spyder相关\DatingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    
    print("you will probably like this person: ",resultList[classifierResult-1])
    
#手写识别系统
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i +j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('F:\Anaconda-spyder相关\TrainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, : ] = img2vector('F:/Anaconda-spyder相关/TrainingDigits/%s' %fileNameStr)
        
    testFileList = listdir('F:\Anaconda-spyder相关\TestDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        
        vectorUnderTest = img2vector('F:/Anaconda-spyder相关/TestDigits/%s' %fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1
    
    print("\nthe total number of error is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))