# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:24:14 2020

@author: Elvis Shehu
"""

import numpy as np
import matplotlib.pyplot as plt
from load_data import load_MNIST
from load_data import load_cifar_10_batches

#get softmax function
def softmax( x, ax=1 ):
    m = np.max( x, axis=ax, keepdims=True )#max per row
    p = np.exp( x - m )
    return ( p / np.sum(p,axis=ax,keepdims=True) )

#get activation function: 'h(a)=tanh(a)', h(a)=cos(a), h(a)=log(1+e^a)
def activation(func):       
    if(func == "tanh"):
        return lambda x: np.tanh(x)
    elif(func == "cos"):        
        return lambda x: np.cos(x)
    elif(func == "log"):
        return lambda x: np.log(1 + np.exp(x))
    else :
        print("Not valid activation function")
        return 
    
#get derivatives of the activation functions
def activation_div(func):       
    if(func == "tanh"):
        return lambda x: 1.0 - np.tanh(x)**2
    elif(func == "cos"):        
        return lambda x: (-1)*np.sin(x)
    elif(func == "log"):
        return lambda x: np.exp(x) / (1 + np.exp(x))  
    else :
        print("Not valid activation function")
        return     

#Calculate the Gradients of Objective function   
def cost_grad_softmax(W1, W2, X, t, func, lamda):     
    h = activation(func)    
    dh =  activation_div(func)       
    #forward propagation
    Z1 = X.dot(W1.T) #(N,M)    
    A1 = h(Z1)# (N,M)    
    y = A1.dot(W2.T)# (N,K)           
    max_error = np.max(y, axis=1)
    s = softmax(y)# (N,K)   
    # Compute the cost function to check convergence
    # Using the logsumexp trick for numerical stability - lec8.pdf slide 43
    Ew = np.sum(t * y) - np.sum(max_error) - \
         np.sum(np.log(np.sum(np.exp(y - np.array([max_error, ] * y.shape[1]).T), 1))) - \
         (0.5 * lamda) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))                 
    #back propagation
    dZ2 = (t - s) 
    dZ1 = np.multiply(np.dot(dZ2, W2), dh(Z1))        
    gradEw2 = np.dot(dZ2.T, A1) - lamda * W2    
    gradEw1 = np.dot(dZ1.T, X)  - lamda * W1
    gradEw = [gradEw1, gradEw2]      
    return Ew, gradEw

#Train the Neural Network by Stochastic Gradient Descent
def fit_model(t, X, func, lamda, W1_init, W2_init, options, batch_size):
    W1 = W1_init
    W2 = W2_init    
    # Maximum number of iteration of gradient ascend
    _iter = options[0]
    # Tolerance
    tol = options[1]
    # Learning rate
    eta = options[2]
    Ewold = -np.inf
    costs = []    
    for i in range( 1, _iter+1 ):        
        for j in range(0, X.shape[0], batch_size):
            xj = X[j:j+batch_size]
            tj = t[j:j+batch_size]                                   
            Ew, gradEw = cost_grad_softmax(W1, W2, xj, tj, func, lamda)                           
            # Update parameters based on gradient ascend
            W1 = W1 + eta * gradEw[0]
            W2 = W2 + eta * gradEw[1]
        # save cost    
        costs.append(Ew)    
        if i % 10 == 0:
            print('Epoch : %d, Cost function :%f' % (i, Ew))
        # Break if you achieve the desired accuracy in the cost function
        if np.abs(Ew - Ewold) < tol:
            break     
        W = [W1, W2]
        Ewold = Ew
    return W, costs

#get the NN predictions
def predict(Weights, X_test):
    h = activation(func)        
    Z1 = X_test.dot(Weights[0].T)
    A1 = h(Z1)   
    Z2 = A1.dot(Weights[1].T)
    y_pred = softmax(Z2)
    pred = np.argmax(y_pred, 1)    
    return pred 

#plot ther error fuction
def plot_error(costs):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(format(options[2], 'f')))
    plt.show() 
    
#Initialize the NN weights by Xavier initialization
def init_network(input_dim, hidden_dim, output_dim):
    #num of classes
    K = output_dim
    #num of hidden units
    M = hidden_dim
    #num of input
    D = input_dim 
    # initialize W1 (M, D) for the gradient ascent 
    W1 = np.random.rand(M, D) * np.sqrt(2/D)
    # initialize W2 (K, M) for the gradient ascent  
    W2 = np.random.rand(K, M) * np.sqrt(2/M)                
    return W1, W2


#convert vector to matrix
def vector_to_matrix(W, w1_shape, w2_shape):
    w1_size = w1_shape[0]*w1_shape[1]    
    W1, W2 = W[:w1_size, ...], W[w1_size:, ...]    
    W1 = np.reshape(W1, w1_shape)
    W2 = np.reshape(W2, w2_shape)
    return W1, W2

#gradient checking function
def gradcheck_softmax(W1init, W2init, X, t, func, lamda):          
    W1 = np.random.rand(*W1init.shape)
    W2 = np.random.rand(*W2init.shape)
    w1_shape = W1.shape
    w2_shape = W2.shape
    
    epsilon = 1e-6    
    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])       
    _, gradEw = cost_grad_softmax(W1, W2, x_sample, t_sample, func, lamda)          
    gradEw = np.concatenate((gradEw[0].flatten(), gradEw[1].flatten()))
    W = np.concatenate((W1.flatten(), W2.flatten()))    
    num_w = W.shape[0]    
    Ew_plus = np.zeros((num_w, 1))
    Ew_minus = np.zeros((num_w, 1))
    numericalGrad = np.zeros(num_w)    
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad    
    for i in range(num_w):            
        #add epsilon to the w            
        w_tmp = np.copy(W)
        w_tmp[i] += epsilon        
        w_1, w_2 = vector_to_matrix(w_tmp, w1_shape, w2_shape)                
        Ew_plus[i], _ = cost_grad_softmax(w_1, w_2, x_sample, t_sample, func, lamda)        
        #subtract epsilon to the w
        w_tmp = np.copy(W)
        w_tmp[i] -= epsilon
        w_1, w_2 = vector_to_matrix(w_tmp, w1_shape, w2_shape)        
        Ew_minus[i], _ = cost_grad_softmax(w_1, w_2, x_sample, t_sample, func, lamda)            
        numericalGrad[i] = (Ew_plus[i] - Ew_minus[i]) / (2 * epsilon)                
    return (gradEw, numericalGrad)


#load dataset
def load_data(file_name):
    if(file_name == "mnist"):
        print("Loading Mnist Dataset.....")
        return load_MNIST()
    elif(file_name == "cifar"):
        print("Loading CIFAR-10 Dataset.....")        
        return load_cifar_10_batches('cifar-10-batches-py')
    else :
        print("Not valid dataset!")
        return

'''load dataset : 'mnist' for the MNIST DATASET , 'cifar' for the CIFAR-10 DATASET'''
X_train, X_test, y_train, y_test = load_data("mnist")
#X_train, X_test, y_train, y_test = load_data("cifar")

#normalize data
X_train = X_train.astype(float)/255
X_test = X_test.astype(float)/255
#add add vector of 1 for biase 
X_train = np.hstack( (np.ones((X_train.shape[0],1) ), X_train) )
X_test = np.hstack( (np.ones((X_test.shape[0],1) ), X_test) )



# Maximum number of iteration of gradient ascend, Tolerance, Learning rate
M = 10
eta = 0.5/X_train.shape[0]
options = [30, 1e-7, 5e-5]

# initialize the model
W1_init, W2_init = init_network(X_train.shape[1], M, y_train.shape[1])
# regularization parameter
lamda = 0.1
#activation functions (log , cos, tanh)
func = "log"
#batch size
batch_size = 100
#Train model
Weights, costs = fit_model(y_train, X_train, func, lamda, W1_init, W2_init, options, batch_size)
#plot train error
plot_error(costs)
#get predictions  
y_pred = predict(Weights, X_test)
#get accuracy
acc = np.mean( y_pred == np.argmax(y_test,1) )
print("accuracy ", acc)
print("Iterations :", options[0], "\nTolerance :", options[1], "\nLearning Rate :", options[2],
      "\nBatch size :", batch_size, "\nlamda :", lamda, "\nfunc :", func, "\nHidden Neurons", M)

#Gradient check
gradEw, numericalGrad = gradcheck_softmax(W1_init, W2_init, X_train, y_train, func, lamda)
print( "The difference estimate for gradient of w is : ", np.max(np.abs(gradEw - numericalGrad)) )



























