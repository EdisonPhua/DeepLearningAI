

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
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    return v


def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    x_norm = np.linalg.norm(x,axis=1,keepdims=True)
    x_normalized = x/x_norm
    return x_normalized
    
    
x = np.array([
    [0, 3, 4],
    [1, 6, 4]])


print(normalizeRows(x))



    


