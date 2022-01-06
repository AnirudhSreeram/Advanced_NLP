from matplotlib.pyplot import axis
import numpy as np
from pdb import set_trace as bp
from abc import ABC, abstractmethod

from numpy.ma import log


class NNComp(ABC):
    """
    This is one design option you might consider. Though it's
    solely a suggestion, and you are by no means required to stick
    to it. You can implement your NN modules as concrete
    implementations of this class, and fill forward and backward
    methods for each module accordingly.
    """
    def __init__(self,inp,neuron, hact, learning_rate=0.1):
        self.neuron = neuron
        self.hact = hact
        self.W = np.random.randn(neuron, inp) # initilise weights
        self.b = np.zeros((neuron, 1))         # initialise bias 
        self.act, self.dif_act = self.activationFunc.get(hact) # initialise activations 
        self.learning_rate = learning_rate    # initialise activation

    #@abstractmethod
    def forward(self, A_prev):
        # compute forward pass for the given layer.
        
        self.A_prev = A_prev
        self.z = np.dot(self.W, self.A_prev) + self.b                # Z = W * x + b 
        self.A = self.act(self.z)                                    # X = activation(Z)
        return self.A

    #@abstractmethod
    def backward(self, dA):
        # Compute the gradients w.r.t the cost function for each layer
        actdz = self.dif_act(self.z)
        _ ,c = self.activationFunc['softmax']     
        if self.dif_act == c:                                        # check if it is the last layer for cross entropy performs differential of both softmax and the loss jointly
            dz = dA
        else:                                                        # else multiply with the dif activation
            dz = np.multiply(actdz,dA)  # dZ = act(Z) * cost_grad
        dw = 1/dz.shape[1] * np.dot(dz,self.A_prev.T)                # dim matching
        db = 1/dz.shape[1] * np.sum(dz, axis=1 , keepdims=True)      # for dim matching
        dA_prev = np.dot(self.W.T, dz)                               # grad = W.T * dZ
        self.W = self.W - self.learning_rate * dw                    # update weights w = w - lr * dw
        self.b = self.b - self.learning_rate * db                    # update bias   b = b - lr * db
        return dA_prev  

    ### helper functions ######
    
    # defining activations and their differentials
    ## Tanh activation
    def tanh(x):
        return np.tanh(x)

    def dif_tanh(x):
        return (1-np.square(np.tanh(x)))
    
    ## Sigmoid activation
    def sigmoid(x):
        return (1/(1+np.exp(-x)))

    def dif_sigmoid(self,x):
        return (self.sigmoid(x) * (1-self.sigmoid(x)))
    
    ## Relu activation function
    def Relu(x):
        return np.clip(x, 0, np.inf)
        #temp = [max(0,value) for value in x]
        #return np.array(temp, dtype=float)
    
    def dif_Relu(z):
        return (z > 0).astype(int)

    def softmax(X):
        sf =[]
        for i in range(X.shape[1]):
            exps = np.exp(X[:,i]- np.max(X[:,i])) # - np.max(X))
            sf.append(exps / np.sum(exps))
        return np.array(sf,dtype=np.float64).T
    
    def dif_softmax(vec):
        # we perform dif of softmax jointly with cross entropy function
        return vec
    

    # defining loss and its derivative
    # Log loss
    def logloss(self,y,pred):
        return (-(y*np.log(pred)+(1-y)*np.log(pred)))
    
    def dif_logloss(self,y,a):
        return ((a-y)/(a*(1-a)))

    # Cross entropy losss function and its derivative
    def cross_entropy_loss(self,vec,Y):
        m = Y.shape[0]
        Y_p = vec
        log_like = - np.log(Y_p[range(m),Y] + .001**10)
        log_like = [0 if x == np.nan else x for x in log_like]
        loss = np.sum(log_like)/m
        return loss
        
    def dif_cross_entropy_loss(self,gr,Y):
        m = Y.shape[0]
        gr[range(m),Y] -= 1
        gr /= m
        return gr    

    # define various actiuvations
    activationFunc ={ 'tanh'    : (tanh,dif_tanh),
                      'sigmoid' : (sigmoid, dif_sigmoid),
                      'relu'    : (Relu, dif_Relu),
                      'softmax': (softmax, dif_softmax)}
     
    
    
    
    


