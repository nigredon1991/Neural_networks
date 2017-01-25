# -*- coding: utf-8 -*-
import numpy as np
import pandas
alfa = 1/100.0 # 
def sigmoid(x):
    return 1/(1+np.exp(-x*alfa))

class NN:
    "neiral network direct propagation(feedforward)"
    def __init__(self, size_in, size_hidden, size_out):
        #initialization
        self.size_in = size_in
        self.size_hidden = size_hidden
        self.size_out = size_out
        #hidden layer
        self.weights0 = np.random.random([size_in,size_hidden])
        self.delta_hidden = np.zeros(size_hidden)  #error for one neiron, and after local gragient
        self.bias_h = np.random.random_sample()
        self.train_values_h = np.zeros(size_hidden)#last calculate out of neiron
        #out layer
        self.weights1 = np.random.random([size_hidden, size_out])
        self.delta_out = np.zeros(size_out)
        self.bias_out = np.random.random_sample()
        self.train_values_out = np.zeros(size_out)
        
        self.E_out = np.zeros(size_out)# Error in last repeat for out neiral network

    def predict(self,X): # X - input example
    #prediction out of NN
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        self.train_values_h = np.dot(self.weights0.T,X)
        self.train_values_h += self.bias_h
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] = sigmoid(self.train_values_h[i])
        
        self.train_values_out = np.dot(self.weights1.T,self.train_values_h)
        self.train_values_out += self.bias_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
        
        return self.train_values_out
        
    def learning(self, X,y, coef=0.1): # X - input example, y - expected yield, coef - coefficient of learning
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
            
        self.train_values_h = np.dot(self.weights0.T,X)
        self.train_values_h += self.bias_h
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] = sigmoid(self.train_values_h[i])
        
        self.train_values_out = np.dot(self.weights1.T,self.train_values_h)
        self.train_values_out += self.bias_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
      
        self.delta_out = -(y - self.train_values_out)
        self.E_out = sum(self.delta_out*self.delta_out/2)
        self.delta_out = self.delta_out  * self.train_values_out* ( 1 - self.train_values_out)
        
        self.delta_hidden = np.dot(self.delta_out,self.weights1.T)
        self.delta_hidden = self.delta_hidden  * self.train_values_h * ( 1- self.train_values_h)
        
        self.weights1 = self.weights1*0.9999 - coef * np.dot(self.train_values_h[:,None],self.delta_out[:,None].T)
        self.weights0 = self.weights0*0.9999 - coef * np.dot(X[:,None],self.delta_hidden[:,None].T)
        return 1

data = pandas.read_csv('train.csv')
y_train = data["label"] # 42000

X_train = data.drop("label", axis = 1) # 42000 * 784
X_train = X_train.as_matrix()
X_train[X_train>0] = 1   #normalization
from time import time
t = time()
#Initialization
MyNN = NN( 784,100,10)
y_temp = np.zeros(10)
#Learning
for i in np.random.randint(42000, size = 30000):
    y_temp[y_train[i]] = 1
    MyNN.learning(X = X_train[i],y = y_temp,coef =0.8)
    y_temp[y_train[i]] = 0

for i in np.random.randint(42000, size = 2000):
    y_temp[y_train[i]] = 1
    MyNN.learning(X = X_train[i],y = y_temp,coef =0.3)
    y_temp[y_train[i]] = 0  
#Verification
acc = 0.0 
num = np.zeros(10)
for i in np.random.randint(42000, size = 1000):
   if(np.argmax(MyNN.predict(X_train[i])) == y_train[i]):
    acc+=1
    num[y_train[i]]+=1

print "accurance:", acc/1000
print num
print "time"
print time()-t
