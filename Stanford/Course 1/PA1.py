import math


def basic_sigmoid(x):
    
    x = 1/(1+math.exp(-x))
    
    return x


print(basic_sigmoid(3))

h



