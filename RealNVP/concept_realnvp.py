import tensorflow as tf 
import numpy as np 


def affine_coupling_block():

    # log Jacobian determinant 在可逆的狀況下，會是"下三角形"，且在對角線之外的數值應該都是0。
    # 因此只需要把log(s)加起來就行了。 
    # ref1. https://proceedings.neurips.cc/paper/2016/hash/ddeebdeefdb7e7e7a697e1c3e3d8ef54-Abstract.html
    # ref2. https://srbittner.github.io/2019/06/26/normalizing_flows/

    pass  

def RealNVP(): 
    x = tf.keras.Input(shape=[28, 28, 1])
    
    pass 

def main(): 
    pass  

if __name__ == "__main__":
    main() 
pass 