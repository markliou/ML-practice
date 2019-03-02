import numpy as np 

def KL_div_with_normal(X):
    ## X with the shape of (n, l)
    X = np.sum(X, axis=0)/X.shape[0]
    return np.sum(X * (np.log(X+1E-9) - np.log(1/X.shape[0])))
pass

a = np.array([[1,0,0],[0,1,0]])
print(KL_div_with_normal(a))
a = np.array([[1,0,0],[1,0,0]])
print(KL_div_with_normal(a))
a = np.array([[1,0,0],[0,1,0],[0,1,0],[1,0,0],[0,0,1]])
print(KL_div_with_normal(a))