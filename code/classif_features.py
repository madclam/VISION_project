# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:07:57 2017

@author: Shiro
"""
import numpy as np
import cv2
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import  KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.mixture import GMM
import time

def get_raw_data(train_file):
    data = []
    label = []
    with open(train_file, 'r') as f:
        for datas in f:
            datas = datas.split()
            data.append((cv2.imread(datas[0], 0)).flatten())
            label.append(datas[1])
    return data, label
    
    
    
    
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
        
        
        
def __init__():
    x_train, y_train = get_features('cifar_train.data', 'cifar_train.solution')
    x_test, y_test = get_features('cifar_test.data', 'cifar_test.solution')
    x_valid, y_valid = get_features('cifar_validation.data', 'cifar_validation.solution')
    
    #x_train, y_train = get_raw_data('train.txt')
    #x_test, y_test = get_raw_data('test.txt') 
    #x_valid, y_valid = get_raw_data('validation.txt') 
    
    print('### Decision Tree ###')
    clf = DecisionTreeClassifier(random_state=0)
    begin = time.time()
    clf.fit(x_train, y_train)
    print('Time computing : ', time.time() - begin)
    clf_pred_test = clf.predict(x_test)
    clf_pred_valid = clf.predict(x_valid)
    print("tree test accuracy :", sklearn.metrics.accuracy_score(clf_pred_test, y_test))
    print("tree validation accuracy :", sklearn.metrics.accuracy_score(clf_pred_valid, y_valid))
    print("tree test recall :", sklearn.metrics.recall_score(clf_pred_test, y_test, average='macro'))
    print("tree validation recall :", sklearn.metrics.recall_score(clf_pred_valid, y_valid, average='macro'))
    print("tree test precision :", sklearn.metrics.precision_score(clf_pred_test, y_test, average='macro'))
    print("tree validation precision :", sklearn.metrics.precision_score(clf_pred_valid, y_valid, average='macro'))
    print("tree test F1 :", sklearn.metrics.f1_score(clf_pred_test, y_test, average='macro'))
    print("tree validation F1 :", sklearn.metrics.f1_score(clf_pred_valid, y_valid, average='macro'))
    
    print('\n### Random Forest ###')
    clf_forest = RandomForestClassifier(max_depth=2, random_state=0, n_estimators= 500)
    begin = time.time()
    clf_forest.fit(x_train, y_train)
    print('Time computing : ', time.time() - begin)
    clf_forest_pred_test = clf_forest.predict(x_test)
    clf_forest_pred_valid = clf_forest.predict(x_valid)
    print("Forest test accuracy :", sklearn.metrics.accuracy_score(clf_forest_pred_test, y_test))
    print("Forest validation accuracy :", sklearn.metrics.accuracy_score(clf_forest_pred_valid, y_valid))
    print("Forest test recall :", sklearn.metrics.recall_score(clf_forest_pred_test, y_test, average='macro'))
    print("Forest validation recall :", sklearn.metrics.recall_score(clf_forest_pred_valid, y_valid, average='macro'))
    print("Forest test precision :", sklearn.metrics.precision_score(clf_forest_pred_test, y_test, average='macro'))
    print("Forest validation precision :", sklearn.metrics.precision_score(clf_forest_pred_valid, y_valid, average='macro'))
    print("Forest test F1 :", sklearn.metrics.f1_score(clf_forest_pred_test, y_test, average='macro'))
    print("Forest validation F1 :", sklearn.metrics.f1_score(clf_forest_pred_valid, y_valid, average='macro'))
    print(len(y_valid))
    print(len(clf_forest_pred_valid))

    print('\n### Naives Bayes Gaussian ###')
    gnb = GaussianNB()
    begin = time.time()
    gnb.fit(x_train, y_train)
    print('Time computing : ', time.time() - begin)
    gnb_pred_test = gnb.predict(x_test)
    gnb_pred_valid = gnb.predict(x_valid)
    print("Bayes Gaussian test accuracy :", sklearn.metrics.accuracy_score(gnb_pred_test, y_test))
    print("Bayes Gaussian validation accuracy :", sklearn.metrics.accuracy_score(gnb_pred_valid, y_valid))
    print("Bayes Gaussian test recall :", sklearn.metrics.recall_score(gnb_pred_test, y_test, average='macro'))
    print("Bayes Gaussian validation recall :", sklearn.metrics.recall_score(gnb_pred_valid, y_valid, average='macro'))
    print("Bayes Gaussian test precision :", sklearn.metrics.precision_score(gnb_pred_test, y_test, average='macro'))
    print("Bayes Gaussian validation precision :", sklearn.metrics.precision_score(gnb_pred_valid, y_valid, average='macro'))
    print("Bayes Gaussian test F1 :", sklearn.metrics.f1_score(gnb_pred_test, y_test, average='macro'))
    print("Bayes Gaussian validation F1 :", sklearn.metrics.f1_score(gnb_pred_valid, y_valid, average='macro'))
    
    print('\n### Naives Bayes Multinomial ###')
    nbm = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    begin = time.time()
    nbm.fit(x_train, y_train)
    print('Time computing : ', time.time() - begin)
    nbm_pred_test = nbm.predict(x_test)
    nbm_pred_valid = nbm.predict(x_valid)
    print("Bayes Multinomial test accuracy :", sklearn.metrics.accuracy_score(nbm_pred_test, y_test))
    print("Bayes Multinomial validation accuracy :", sklearn.metrics.accuracy_score(nbm_pred_valid, y_valid))
    print("Bayes Multinomial test recall :", sklearn.metrics.recall_score(nbm_pred_test, y_test, average='macro'))
    print("Bayes Multinomial validation recall :", sklearn.metrics.recall_score(nbm_pred_valid, y_valid, average='macro'))
    print("Bayes Multinomial test precision :", sklearn.metrics.precision_score(nbm_pred_test, y_test, average='macro'))
    print("Bayes Multinomial validation precision :", sklearn.metrics.precision_score(nbm_pred_valid, y_valid, average='macro'))
    print("Bayes Multinomial test F1 :", sklearn.metrics.f1_score(nbm_pred_test, y_test, average='macro'))
    print("Bayes Multinomial validation F1 :", sklearn.metrics.f1_score(nbm_pred_valid, y_valid, average='macro'))
    
    print('\n### SVM rbf ###')
    clf_svm = SVC(C=1.0, kernel = 'rbf')
    begin = time.time()
    clf_svm.fit(x_train, y_train)
    print('Time computing : ', time.time() - begin)
    clf_svm_pred_test = clf_svm.predict(x_test)
    clf_svm_pred_valid = clf_svm.predict(x_valid)
    print("SVM Kernel test accuracy :", sklearn.metrics.accuracy_score(clf_svm_pred_test, y_test))
    print("SVM Kernel validation accuracy :", sklearn.metrics.accuracy_score(clf_svm_pred_valid, y_valid))
    print("SVM Kernel test recall :", sklearn.metrics.recall_score(clf_svm_pred_test, y_test, average='macro'))
    print("SVM Kernel validation recall :", sklearn.metrics.recall_score(clf_svm_pred_valid, y_valid, average='macro'))
    print("SVM Kernel test precision :", sklearn.metrics.precision_score(clf_svm_pred_test, y_test, average='macro'))
    print("SVM Kernel validation precision :", sklearn.metrics.precision_score(clf_svm_pred_valid, y_valid, average='macro'))
    print("SVM Kernel test F1 :", sklearn.metrics.f1_score(clf_svm_pred_test, y_test, average='macro'))
    print("SVM Kernel validation F1 :", sklearn.metrics.f1_score(clf_svm_pred_valid, y_valid, average='macro'))
    
    print('\n### SVM Linear ###')
    clf_svm = SVC(C=1.0, kernel = 'linear')
    begin = time.time()
    clf_svm.fit(x_train, y_train)
    print('Time computing : ', time.time() - begin)
    clf_svm_pred_test = clf_svm.predict(x_test)
    clf_svm_pred_valid = clf_svm.predict(x_valid)
    print("SVM Linear test accuracy :", sklearn.metrics.accuracy_score(clf_svm_pred_test, y_test))
    print("SVM Linear validation accuracy :", sklearn.metrics.accuracy_score(clf_svm_pred_valid, y_valid))
    print("SVM Linear test recall :", sklearn.metrics.recall_score(clf_svm_pred_test, y_test, average='macro'))
    print("SVM Linear validation recall :", sklearn.metrics.recall_score(clf_svm_pred_valid, y_valid, average='macro'))
    print("SVM Linear test precision :", sklearn.metrics.precision_score(clf_svm_pred_test, y_test, average='macro'))
    print("SVM Linear validation precision :", sklearn.metrics.precision_score(clf_svm_pred_valid, y_valid, average='macro'))
    print("SVM Linear test F1 :", sklearn.metrics.f1_score(clf_svm_pred_test, y_test, average='macro'))
    print("SVM Linear validation F1 :", sklearn.metrics.f1_score(clf_svm_pred_valid, y_valid, average='macro'))
    
    print('\n### SVM Poly ###')
    clf_svm = SVC(C=1.0, kernel = 'poly', degree=2)
    begin = time.time()
    clf_svm.fit(x_train, y_train)
    print('Time computing : ', time.time() - begin)
    clf_svm_pred_test = clf_svm.predict(x_test)
    clf_svm_pred_valid = clf_svm.predict(x_valid)
    print("SVM Poly test accuracy :", sklearn.metrics.accuracy_score(clf_svm_pred_test, y_test))
    print("SVM Poly validation accuracy :", sklearn.metrics.accuracy_score(clf_svm_pred_valid, y_valid))
    print("SVM Poly test recall :", sklearn.metrics.recall_score(clf_svm_pred_test, y_test, average='macro'))
    print("SVM Poly validation recall :", sklearn.metrics.recall_score(clf_svm_pred_valid, y_valid, average='macro'))
    print("SVM Poly test precision :", sklearn.metrics.precision_score(clf_svm_pred_test, y_test, average='macro'))
    print("SVM Poly validation precision :", sklearn.metrics.precision_score(clf_svm_pred_valid, y_valid, average='macro'))
    print("SVM Poly test F1 :", sklearn.metrics.f1_score(clf_svm_pred_test, y_test, average='macro'))
    print("SVM Poly validation F1 :", sklearn.metrics.f1_score(clf_svm_pred_valid, y_valid, average='macro'))
    

__init__()