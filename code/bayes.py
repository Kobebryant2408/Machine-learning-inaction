# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:24:44 2017

@author: 晟玮

主要内容：区分垃圾邮件
"""

import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], 
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'gare'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
        
    return list(vocabSet)

#根据单词出现次数构造数据向量
def setOfWords2Vec(vocabList, inputSet):#词集模型，根据单次是否出现得到
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!"% word)
    
    return returnVec

#训练算法得到模型参数
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    
    return p0Vect, p1Vect, pAbusive  #通过训练得到的三个重要数据，就像决策树算法通过训练得到的决策树一样重要

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print("testEntry classified as: %d" %classifyNB(thisDoc, p0V, p1V, pAb))
    
def bagOfWordsVecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    
    return returnVec

#解析文本的函数，对于正则表达式这一块掌握不够
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    
    for i in range(1,26):
        wordList = textParse(open('F:/Anaconda-spyder相关/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('F:/Anaconda-spyder相关/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    
    vocabList = createVocabList(docList)
    
    trainingSet = list(range(50))
    testSet = []
    
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
        
    trainMat = []
    trainingClasses = []
    
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainingClasses.append(classList[docIndex])
        
    p0V, p1V, pSpam = trainNB0(trainMat, trainingClasses)
    errorCount = 0
    
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam)!=classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])#没有%，也没有format
   # a = float(errorCount)/len(testSet)
   # print("the error rate is: %f" % (float(errorCount)/len(testSet))) #符号'/'对格式化输出有什么影响啊
    print('the error rate is: ',float(errorCount)/len(testSet)) 
    
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
        
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse = True)
        
    return sortedFreq[:30]
    
def localWords(feed1, feed0):
    import feedparser#这个库的应用要熟悉;使用指令conda install (feedparser)安装别的库
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
            
    trainingSet = list(range(2*minLen))
    testSet = []
    
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    trainMat = []
    trainClasses = []
    
    for docIndex in trainingSet:
        trainMat.append(bagOfWordsVecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
        
    p0V, p1V, pSpam = trainNB0(trainMat, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWordsVecMN(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -5:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -5:
            topNY.append((vocabList[i], p1V[i]))
            
    sortedSF = sorted(topSF, key = lambda pair:pair[1], reverse = True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF:
        print(item[0])
        
    sortedNY = sorted(topNY, key = lambda pair:pair[1], reverse = True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sortedNY:
        print(item[0])