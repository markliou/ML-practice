#!/usr/bin/python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc as sm
import tensorflow_datasets as tfds 
import jax 
from jax.experimental import optimizers
import numpy as np

global prngkey
prngkey = jax.random.PRNGKey(20) 

def G():
    pass 

def D():
    pass 

def CNN_init():
    pass 
    
def conv2d():
    pass 

def transpose_conv2d():
    jax.lax.conv_general_dilated(np.zeros([1,1,1,1]), 
                                 np.ones([3,3,1,5]), 
                                 (1,1), # strides
                                 ((2,2),(2,2)), # padding
                                 [2,2], # feature map dialation
                                 [1,1], # kernel dialation
                                 dimension_numbers=('NHWC','HWIO','NHWC') # convolution mode
                                 )
pass 

def cross_entropy():
    pass 

def main():
    batch_size = 32
    G_arch = [] 
    
    ## Import MNIST data
    dataset = tfds.load('mnist',  shuffle_files=True)
    tr, ts = iter(dataset['train'].batch(128).prefetch(1).repeat()), iter(dataset['test'].batch(1).repeat())


    
pass
    
    
    
if __name__=='__main__':
    main()
pass

