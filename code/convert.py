# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:40:16 2017

@author: Shiro
"""
import numpy as np
def convert_label(file):
    f = open(file, 'r')
    data = [line.replace('\n','') for line in f ]
    print(len(data))
    f.close()
    f = open(file.replace('cifar', 'cifar10'), 'w')
    for i in range(0, len(data)):
        temp = np.zeros((10))
        temp[int(data[i])] = 1
        for cpt, lab in enumerate(temp):
            f.write(str(int(lab)))
            if cpt != 9:
                f.write(' ')
        f.write('\n')
    f.close()
    
def main():
    convert_label('cifar_test.solution')
    
main()