# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:44:33 2017

@author: 晟玮
"""

import KNN2 
import matplotlib.pyplot as plt
import numpy as np

datingDataMat, datingLabels = KNN2.file2matrix('F:\Anaconda-spyder相关\DatingTestSet2.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[ : ,0], datingDataMat[ : ,1], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))

plt.show()