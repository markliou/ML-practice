#!/usr/bin/python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sm


def G(Z,gw):
## this function is setted to generate a 28*28 digit picture
## from a normal distribution z
## here, z with 32 channels is setted to size of 7*7 which can be transpose convoluted 
## 2 times with stride 2 to get 28*28 matrix
    # the shape of z should be [-1, 7, 7, 32]
    batch_size = tf.shape(Z)[0]
    g_conv1 = tf.nn.conv2d_transpose(      Z, gw['w1'], [batch_size, 14, 14, 128], [1, 2, 2, 1]) + gw['b1'] # 14*14, 128 channels
    g_conv1 = tf.nn.relu(g_conv1)
    g_conv2 = tf.nn.conv2d_transpose(g_conv1, gw['w2'], [batch_size, 14, 14, 256], [1, 1, 1, 1]) + gw['b2'] # 14*14, 256 channels
    g_conv2 = tf.nn.relu(g_conv2)
    g_conv3 = tf.nn.conv2d_transpose(g_conv2, gw['w3'], [batch_size, 28, 28, 512], [1, 2, 2, 1]) + gw['b3'] # 28*28, 512 channels
    g_conv3 = tf.nn.relu(g_conv3)
    g_conv4 = tf.nn.conv2d_transpose(g_conv3, gw['w4'], [batch_size, 28, 28, 256], [1, 1, 1, 1]) + gw['b4'] # 28*28, 256 channels
    g_conv4 = tf.nn.relu(g_conv4)
    g_conv5 = tf.nn.conv2d_transpose(g_conv4, gw['w5'], [batch_size, 28, 28,   1], [1, 1, 1, 1]) + gw['b5'] # 28*28, 1 channels
    x = tf.nn.sigmoid(g_conv5)
    return x
    pass

def D(x, dw):
## this function is supposeed to discriminate the x is from 
## ground truth or generator
    d_conv1 = tf.nn.conv2d(      x, dw['w1'], [1, 2, 2, 1], 'SAME') + dw['b1'] # 14*14, 512 channels 
    d_conv1 = tf.nn.relu(d_conv1)
    d_conv2 = tf.nn.conv2d(d_conv1, dw['w2'], [1, 2, 2, 1], 'SAME') + dw['b2'] # 7*7, 128 channels 
    d_conv2 = tf.nn.relu(d_conv2)
    d_conv3 = tf.nn.conv2d(d_conv2, dw['w3'], [1, 2, 2, 1], 'SAME') + dw['b3'] # 4*4, 512 channels 
    d_conv3 = tf.nn.relu(d_conv3)
    d_conv4 = tf.nn.conv2d(d_conv3, dw['w4'], [1, 2, 2, 1], 'SAME') + dw['b4'] # 2*2, 256 channels 
    d_conv4 = tf.nn.relu(d_conv4)
    d_conv5 = tf.nn.conv2d(d_conv4, dw['w5'], [1, 1, 1, 1], 'VALID') + dw['b5'] # 1*1, 1 channels 
    d_conv5 = tf.nn.sigmoid(d_conv5)
    return tf.reshape(d_conv5, [-1,])
    pass
    
def W():
    initializer = tf.contrib.layers.xavier_initializer()
    gw = {
        'w1' : tf.Variable(initializer([3, 3, 128, 32])),
        'b1' : tf.Variable(initializer([128])),
        'w2' : tf.Variable(initializer([3, 3, 256, 128])),
        'b2' : tf.Variable(initializer([256])),
        'w3' : tf.Variable(initializer([3, 3, 512, 256])),
        'b3' : tf.Variable(initializer([512])),
        'w4' : tf.Variable(initializer([3, 3, 256, 512])),
        'b4' : tf.Variable(initializer([256])),
        'w5' : tf.Variable(initializer([3, 3,   1, 256])),
        'b5' : tf.Variable(initializer([1])),
    }
    dw = {
        'w1' : tf.Variable(initializer([3, 3, 1, 512])),
        'b1' : tf.Variable(initializer([512])),
        'w2' : tf.Variable(initializer([3, 3, 512, 128])),
        'b2' : tf.Variable(initializer([128])),
        'w3' : tf.Variable(initializer([3, 3, 128, 512])),
        'b3' : tf.Variable(initializer([512])),
        'w4' : tf.Variable(initializer([3, 3, 512, 256])),
        'b4' : tf.Variable(initializer([256])),
        'w5' : tf.Variable(initializer([2, 2, 256, 1])),
        'b5' : tf.Variable(initializer([1])),
    }
    return [gw,dw]
    pass 
    
def main():
    batch_size = 32
    training_iter = 5000
    
    ## Import MNIST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    ## build GAN
    # announce the essential placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1]) # put the real samples
    Y = tf.placeholder(dtype=tf.float32, shape=[None,]) # the real and fake labels
    Z = tf.placeholder(dtype=tf.float32, shape=[None, 7, 7, 32]) # the noise for generating samples
    # announce the weights of G and D
    [gw, dw] = W()
    
    # get samples from generator and discriminator
    gX = G(Z, gw)
    
    # define the loss and optimizers, real samples are labeled 1 and fake samples are labeled 0
    logits_4D = D(tf.concat([gX, X], axis=0), dw) #label: (fake, real)
    logits_4G = D(gX, dw)
    loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits_4D))
    loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G))
    opt_D = tf.train.AdamOptimizer(1e-5).minimize(loss_D)
    opt_G = tf.train.AdamOptimizer(1e-5).minimize(loss_G)
    
    ## start training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iter in range(training_iter):
            x, _ = mnist.train.next_batch(batch_size)
            x = x.reshape([-1, 28, 28, 1])
            y = np.hstack([np.zeros([batch_size,]), np.ones([batch_size,])]) #(fake, real)
            z = np.random.normal(size=batch_size* 7 * 7 * 32).reshape([-1, 7, 7, 32])
            
            # cX, Closs_D, Closs_G, _, _ = sess.run([gX, loss_D, loss_G, opt_D, opt_G], feed_dict={X:x, Y:y, Z:z}) # strategy 1: update simultaneously
            if iter % 500 == 0 : # stragegy 2: make D stroger but update less times
                for D_iter in range(100):
                    cX, Closs_D, Closs_G, _ = sess.run([gX, loss_D, loss_G, opt_D], feed_dict={X:x, Y:y, Z:z})
                    z = np.random.normal(size=batch_size* 7 * 7 * 32).reshape([-1, 7, 7, 32])
            else:
                cX, Closs_D, Closs_G, _ = sess.run([gX, loss_D, loss_G, opt_G], feed_dict={X:x, Y:y, Z:z})
            
            print('iteration:{} loss_D:{} loss_G:{}'.format(iter, Closs_D, Closs_G))
            if iter%10 == 0 :
                visg = cX[0].T
                visr = x[0].T
                visgp = np.vstack([visg, visg, visg]).T
                visgp *= 256
                visrp = np.vstack([visr, visr, visr]).T
                visrp *= 256
                visgp.astype(np.uint8)
                sm.imsave('g.jpg',visgp)
                visrp.astype(np.uint8)
                sm.imsave('r.jpg',visrp)
        
    
    pass
    
    
    
if __name__=='__main__':
    main()
    pass