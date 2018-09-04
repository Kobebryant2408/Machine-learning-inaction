# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:12:39 2018

@author: 晟玮

主要内容：从疝气病症预测病马的死亡率(同logRegres.py)
主要方法：首先loadDataSet加载训练数据集,
         然后数据集导入adaBoostTrainDS训练得到弱分类器集合,
         最后loadDataSet加载测试数据集导入adaClassify得到预测结果,
         预测结果可以和测试集中的结果进行比较分析错误率等
"""

import numpy as np

def loadSimpData():
    dataMat = np.matrix([[1. , 2.1],
                      [2. , 1.1],
                      [1.3, 1. ],
                      [1. , 1. ],
                      [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat,classLabels

#通过阈值(threshVal)比较对数据进行分类(1,-1) ;基于单层决策树的分类器
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data .dimen:特征列;threshVal:特征列的比较值
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#构建弱分类器--单层决策树(最优列，最优阈值，最优比较方法)
def buildStump(dataArr,classLabels,D):#D:每个样本的权值
    dataMatrix = np.mat(dataArr); 
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; 
    bestStump = {}; 
    bestClasEst = np.mat(np.zeros((m,1)))#初始化训练后的最优结果集
    minError = np.inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions 遍历所有特征列
        rangeMin = dataMatrix[:,i].min(); 
        rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps #计算步长
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()#最优预测结果
                    bestStump['dim'] = i              #最优特征列
                    bestStump['thresh'] = threshVal   #最优阈值
                    bestStump['ineq'] = inequal       #最优比较方法(大于or小于)
    return bestStump,minError,bestClasEst

#adaboost算法训练函数
def adaBoostTrainDS(dataArr,classLabels,numIt=40):#numIt:实例数即弱分类器个数;DS指单层决策树(实际应用中可以为别的分类器)
    weakClassArr = []#初始化弱分类器集合
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)   #init D to all equal
    aggClassEst = np.mat(np.zeros((m,1)))#初始化预测分类结果
    for i in range(numIt):#循环求出每个最优单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D.T计算最优单层决策树的权值alpha
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T计算下一次迭代的样本权值向量D
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = np.multiply(D,np.exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))#sign作用:大于0的返回1，小于0的返回-1，等于0的返回0
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        if errorRate == 0.0: 
            break
    return weakClassArr,aggClassEst

#adaboost算法分类器
def adaClassify(datToClass,classifierArr):#待分类样本以及弱分类器集合
    dataMatrix = np.mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return np.sign(aggClassEst)

#从文本文件中加载数据
def loadDataSet(fileName):      #general function to parse tab -delimited floats fileName=F:/Anaconda-spyder相关/Ch07/horseColicTraining2.txt
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#绘制ROC曲线并计算AUC
def plotROC(predStrengths, classLabels):#preStrtengths:预测强度(对于adaboost即aggClassEst的值)
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor 绘制光标的位置
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas); 
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:#tolist为数组或矩阵转列表的函数 
        if classLabels[index] == 1.0:
            delX = 0; 
            delY = yStep;
        else:
            delX = xStep; 
            delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)绘制ROC曲线
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    #绘制随机猜测曲线
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); 
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: ",ySum*xStep)

