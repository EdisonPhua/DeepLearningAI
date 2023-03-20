

import numpy as np  

def sigmoid(x):
    
    s = 1/(1+np.exp(-x))

    return s

def sigmoid_derivative(x):
    
    
    s = x*(1-x)
    return s


def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    




x = np.array([1,2,3])
s=sigmoid(x)
print(sigmoid_derivative(s))




    


