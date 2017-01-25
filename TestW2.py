# -*- coding: utf-8 -*-
import numpy as np
import pandas
alfa = 1/10.0 # 
def sigmoid(x):
    return 1/(1+np.exp(-x*alfa))

class NN:
    "neiral network direct propagation(feedforward)"
    def __init__(self, size_in, size_hidden0,size_hidden1, size_out):
        #initialization
        self.size_in = size_in
        self.size_hidden0 = size_hidden0
        self.size_hidden1 = size_hidden1
        self.size_out = size_out
        #hidden layers0         
        self.weights0 = np.random.random([size_in,size_hidden0])
        self.delta_hidden0 = np.zeros(size_hidden0)  #error for one neiron, and after local gragient
        self.bias_h1 = np.random.random_sample()
        self.train_values_h0 = np.zeros(size_hidden0)#last calculate out of neiron
        #hidden layers1
        self.weights1 = np.random.random([size_hidden0,size_hidden1])
        self.delta_hidden1 = np.zeros(size_hidden1)  #error for one neiron, and after local gragient
        self.bias_h0 = np.random.random_sample()
        self.train_values_h1 = np.zeros(size_hidden1)#last calculate out of neiron
        #out layer
        self.weights_out = np.random.random([size_hidden1, size_out])
        self.delta_out = np.zeros(size_out)
        self.bias_out = np.random.random_sample()
        self.train_values_out = np.zeros(size_out)
        
        self.E_out = np.zeros(size_out)# Error in last repeat for out neiral network

    def predict(self,X): # X - input example
    #prediction out of NN
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        self.train_values_h0 = np.dot(self.weights0.T,X)
        self.train_values_h0 += self.bias_h0
        for i in range(0,self.size_hidden0 ):
            self.train_values_h0[i] = sigmoid(self.train_values_h0[i])

        self.train_values_h1 = np.dot(self.weights1.T,self.train_values_h0)
        self.train_values_h1 += self.bias_h1
        for i in range(0,self.size_hidden1 ):
            self.train_values_h1[i] = sigmoid(self.train_values_h1[i])
        
        self.train_values_out = np.dot(self.weights_out.T,self.train_values_h1)
        self.train_values_out += self.bias_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
        
        return self.train_values_out
        
    def learning(self, X,y, coef=0.1): # X - input example, y - expected yield, coef - coefficient of learning
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
            
        self.train_values_h0 = np.dot(self.weights0.T,X)
        self.train_values_h0 += self.bias_h0
        for i in range(0,self.size_hidden0 ):
            self.train_values_h0[i] = sigmoid(self.train_values_h0[i])

        self.train_values_h1 = np.dot(self.weights1.T,self.train_values_h0)
        self.train_values_h1 += self.bias_h1
        for i in range(0,self.size_hidden1 ):
            self.train_values_h1[i] = sigmoid(self.train_values_h1[i])
            
        self.train_values_out = np.dot(self.weights_out.T,self.train_values_h1)
        self.train_values_out += self.bias_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
      
        self.delta_out = -(y - self.train_values_out)
        self.E_out = sum(self.delta_out*self.delta_out/2)
        self.delta_out = self.delta_out  * self.train_values_out* ( 1 - self.train_values_out)
        
        self.delta_hidden1 = np.dot(self.delta_out,self.weights_out.T)
        self.delta_hidden1 = self.delta_hidden1  * self.train_values_h1 * ( 1- self.train_values_h1)
        
        self.delta_hidden0 = np.dot(self.delta_hidden1,self.weights1.T)
        self.delta_hidden0 = self.delta_hidden0  * self.train_values_h0 * ( 1- self.train_values_h0)
        
        self.weights_out = self.weights_out*0.9999 - coef * np.dot(self.train_values_h1[:,None],self.delta_out[:,None].T)
        self.weights1 = self.weights1*0.9999 - coef * np.dot(self.train_values_h0[:,None],self.delta_hidden1[:,None].T)
        self.weights0 = self.weights0*0.9999 - coef * np.dot(X[:,None],self.delta_hidden0[:,None].T)
        return 1

data = pandas.read_csv('train.csv')
y_train = data["label"] # 42000

X_train = data.drop("label", axis = 1) # 42000 * 784
X_train = X_train.as_matrix()
X_train[X_train>0] = 1   #normalization
from time import time
t = time()
#Initialization
MyNN = NN( 784,100,50,10)
y_temp = np.zeros(10)
#Learning
for i in np.random.randint(42000, size = 34000):
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
sum = np.zeros(10)
for i in np.random.randint(42000, size = 1000):
    sum[y_train[i]]+= 1
    if(np.argmax(MyNN.predict(X_train[i])) == y_train[i]):
        acc+=1
        num[y_train[i]]+=1

print "accurance:", acc/1000
print num/sum
print "time"
print time()-t
