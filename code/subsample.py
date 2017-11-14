# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:18:39 2017

@author: Shiro
"""
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split

def get_features(datas, labels):
    f = open(datas, 'r')
    f2 = open(labels, 'r')
    X = []
    for data in f:
        data = data.split()
        temp = np.zeros((len(data)))
        for i in range(0, len(temp)):
            temp[i] = float(data[i])
        X.append(temp)
    Y = []
    for data in f2:
        data = data.split()
        temp = np.zeros((len(data)))
        for i in range(0, len(temp)):
            temp[i] = float(data[i])
        Y.append(temp)
    return np.asarray(X), np.asarray(Y)

def save_in_file(x, y, namex, namey):
    f = open(namex, 'w')
    for feat in x:
        for cpt, i in enumerate(feat):
            f.write(str(i))
            if(cpt != 255):
                f.write(' ')
            else:
                f.write('\n')
    f.close()
    
    f = open(namey, 'w')
    for feat in y:
        for cpt, i in enumerate(feat):
            f.write(str(int(i)))
            if(cpt != 9):
                f.write(' ')
            else:
                f.write('\n')
                
def subsample(file):
    x , y = get_features('cifar10_train2.data', 'cifar10_train2.solution')
    x_train = x[0:1000]
    y_train = y[0:1000]
    save_in_file(x_train, y_train, 'cifar10_train.data', 'cifar10_train.solution')
    x_test = x[1000:2000]
    y_test = y[1000:2000]
    save_in_file(x_test, y_test, 'cifar10_test.data', 'cifar10_test.solution')
    x_validation = x[2000:3000]
    y_validation = y[2000:3000]
    save_in_file(x_validation, y_validation, 'cifar10_validation.data', 'cifar10_validation.solution')
    
subsample('bg')
    
        