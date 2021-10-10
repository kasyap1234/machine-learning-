import numpy as np 



class LinearRegression(): 
    def __init__(self,L,iter=1000): 
        self.weights=None 
        self.L=L   # L is the learning rate 
        self.iter=1000
        self.intercept=None 
    def fit(self,x,y): 
        n,features=x.shape()
        self.weights=np.zeros(features)
        self.intercept=0
        
        for i in range(self.iter): 
            ypred=np.dot(x,self.weights) + self.intercept 
            dw=(1/n)*np.dot(x.T,(ypred-y))
            dl=(1/n)*np.sum(ypred-y)
            self.weights=self.weights-self.L*dw
            self.intercept=self.intercept -self.L*dl         
     
    def predict(self,x): 
        ypred=np.dot(x,self.weights) + self.intercept 
        return ypred 
    
