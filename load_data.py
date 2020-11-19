# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:28:54 2020

@author: Elvis Shehu
"""
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle
import os


def load_MNIST():
    """
    Load the MNIST dataset. Reads the training and testing files and create matrices.
    :Expected return:
    train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    y_train: the matrix consisting of one 
                        hot vectors on each row(ground truth for training)
    y_test: the matrix consisting of one
                        hot vectors on each row(ground truth for testing)
    """
    
    #load the train files
    df = None    
    y_train = []
    for i in range( 10 ):
        tmp = pd.read_csv( 'mnistdata/train%d.txt' % i, header=None, sep=" " )
        #build labels - one hot vector
        hot_vector = [ 1 if j == i else 0 for j in range(0,10) ]        
        for j in range( tmp.shape[0] ):
            y_train.append( hot_vector )
        #concatenate dataframes by rows    
        if i == 0:
            df = tmp
        else:
            df = pd.concat( [df, tmp] )
    train_data = df.as_matrix()
    y_train = np.array( y_train )    
    #load test files
    df = None    
    y_test = []

    for i in range( 10 ):
        tmp = pd.read_csv( 'mnistdata/test%d.txt' % i, header=None, sep=" " )
        #build labels - one hot vector
        
        hot_vector = [ 1 if j == i else 0 for j in range(0,10) ]
        
        for j in range( tmp.shape[0] ):
            y_test.append( hot_vector )
        #concatenate dataframes by rows    
        if i == 0:
            df = tmp
        else:
            df = pd.concat( [df, tmp] )
    test_data = df.as_matrix()
    y_test = np.array( y_test )    
    return train_data, test_data, y_train, y_test



def unpickle(filename):    
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

#load cifar dataset
def load_cifar_10_batches(file):    
    x = []
    y = []
    for b in range(1,6):
        filename = os.path.join(file, 'data_batch_%d' % (b, ))
        X, Y = unpickle(filename)        
        x.append(X)
        y.append(Y)
    train_data = np.concatenate(x)
    y_train = np.concatenate(y)    
    y_train = np.eye(10)[y_train]    
    test_data, y_test = unpickle(os.path.join(file, 'test_batch'))
    y_test = np.eye(10)[y_test]
    return train_data, test_data, y_train, y_test


 








































