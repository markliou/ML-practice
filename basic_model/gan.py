#!/usr/bin/python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplot.pyplot as plt


def G(z,gw):
## this function is setted to generate a 28*28 digit picture
## from a normal distribution z
## here, z with 32 channels is setted to size of 7*7 which can be transpose convoluted 
## 2 times with stride 2 to get 28*28 matrix
    # the shape of z should be [-1, 7, 7, 32]
    g_conv1 = tf.nn.conv2d_transpose(    z, gw['w1'], [-1, 14, 14, 128], [1, 2, 2, 1]) + gw['b1'] # 14*14, 128 channels
    g_conv1 = tf.nn.relu(g_conv1)
    g_conv2 = tf.nn.conv2d_transpose(conv1, gw['w2'], [-1, 28, 28, 1],  [1, 2, 2, 1]) + gw['b2'] # 28*28, 1 channels
    x = tf.nn.sigmoid(g_conv2)
    return x
    pass

def D(x, dw):
## this function is supposeed to discriminate the x is from 
## ground truth or generator
    d_conv1 = tf.nn.conv2d(      x, dw['w1'], [1, 2, 2, 1], 'SAME') + dw['b1']) # 14*14, 512 channels 
    d_conv1 = tf.nn.relu(d_conv1)
    d_conv2 = tf.nn.conv2d(d_conv1, dw['w2'], [1, 2, 2, 1], 'SAME') + dw['b2']) # 7*7, 128 channels 
    d_conv2 = tf.nn.relu(d_conv2)
    d_conv3 = tf.nn.conv2d(d_conv2, dw['w3'], [1, 2, 2, 1], 'SAME') + dw['b3']) # 4*4, 512 channels 
    d_conv3 = tf.nn.relu(d_conv3)
    d_conv4 = tf.nn.conv2d(d_conv3, dw['w3'], [1, 2, 2, 1], 'SAME') + dw['b4']) # 2*2, 256 channels 
    d_conv4 = tf.nn.relu(d_conv4)
    d_conv5 = tf.nn.conv2d(d_conv4, dw['w3'], [1, 2, 2, 1], 'SAME') + dw['b5']) # 2*2, 1 channels 
    d_conv5 = tf.nn.sigmoid(d_conv5)
    return tf.reshape(d_conv5, [1,])
    pass
    
def W():
    initializer = tf.contrib.layers.xavier_initializer()
    gw = {
        'w1' = initializer([3, 3, 128, 32]),
        'b1' = initializer([128]),
    }
    dw = {
        'w1' = initializer([3, 3, 1, 512]),
        'b1' = initializer([512]),
        'w2' = initializer([3, 3, 512, 128]),
        'b2' = initializer([128]),
        'w3' = initializer([3, 3, 128, 512]),
        'b3' = initializer([512]),
        'w4' = initializer([3, 3, 512, 256]),
        'b4' = initializer([256]),
        'w5' = initializer([3, 3, 256, 1]),
        'w5' = initializer([1]),
    }
    return (gw,dw)
    pass 
    
def main():
    batch_size = 32
    training_iter = 5000
    
    ## Import MNIST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    ## build GAN
    # announce the essential placeholders
    X = tf.placeholder(dtype=tf.float32, [None, 28, 28, 1]) # put the real samples
    Y = tf.placeholder(dtype=tf.float32, [None,]) # the real and fake labels
    Z = tf.placeholder(dtype=tf.float32, [None, 7, 7, 32]) # the noise for generating samples
    # announce the weights of G and D
    gw, dw = W()
    
    # get samples from generator and discriminator
    gX = G(Z, gw)
    
    # define the loss and optimizers, real samples are labeled 1 and fake samples are labeled 0
    logits_4D = D(tf.concat([gX, X], axis=0), dw) #label: (fake, real)
    loss_D = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits_4D)
    logits_4G = D(gX, dw)
    loss_G = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G)
    opt_D = tf.train.AdamOptimizer(0.001).minimize(loss_D)
    opt_G = tf.train.AdamOptimizer(0.001).minimize(loss_G)
    
    ## start training
    with tf.session() as sess:
        sess.run(tf.global_variables_initializer())
        for iter in range(training_iter):
            x, _ = mnist.train.next_batch(batch_size)
            x = x.reshape([-1, 28, 28, 1])
            y = np.hstack(np.zeros([batch_size,]), np.ones([batch_size,])) #(fake, real)
            z = np.random.normal(size=batch_size* 7 * 7 * 32).reshape([-1, 7, 7, 32])
            sess.run([opt_D, opt_G], feed_dict={X:x, Y:y, Z:z})
        
    
    pass
    
    
    
if __name__=='__main__':
    main()
    pass