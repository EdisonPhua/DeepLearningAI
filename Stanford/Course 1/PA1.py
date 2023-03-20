

import numpy as np  

def sigmoid(x):
    
    s = 1/(1+np.exp(-x))

    return s

def sigmoid_derivative(x):
    
    
    s = x*(1-x)
    return s


x = np.array([1,2,3])
s=sigmoid(x)
print(sigmoid_derivative(s))




    


