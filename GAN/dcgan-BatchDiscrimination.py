#!/usr/bin/python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# import matplotlib.pyplot as plt
import scipy.misc as sm

def lrelu(x, lamb = 0.2):
    return tf.maximum(x, x*lamb)
pass

def mini_batch_discrimonation(sample_no, noize_dim, gw):
    # Here will generat new samples for batch discriminator.
    # comparing to iteration run all the samples, this just generats two times and 
    # simplily compare them once to get roughtly estimation.
    return  tf.reduce_mean(
                tf.losses.cosine_distance(
                    labels      = G(np.random.uniform(size = sample_no * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim]), gw),
                    predictions = G(np.random.uniform(size = sample_no * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim]), gw),
                    axis = -1
                )
            )
pass

def G(Z,gw):
## this function is setted to generate a 28*28 digit picture
## from a normal distribution z
## here, z with 32 channels is setted to size of 7*7 which can be transpose convoluted 
## 2 times with stride 2 to get 28*28 matrix

    activation_function = tf.nn.elu
    # activation_function = lrelu
    dropout_keep_rate = 0.5
    
    Z = tf.convert_to_tensor(Z)
    Z = tf.cast(Z, tf.float32)
    
    # the shape of z should be [-1, 1, 1, 128]
    batch_size = tf.shape(Z)[0]
    
    # Z = tf.nn.tanh(Z)
    #Z = tf.layers.batch_normalization(Z)
    g_conv1 = tf.nn.conv2d_transpose(      Z, gw['w1'], [batch_size,  2,  2, 256], [1, 2, 2, 1], padding="VALID") + gw['b1'] # 2*2, 128 channels
    g_conv1 = activation_function(g_conv1)
    # g_conv1 = tf.nn.dropout(g_conv1, dropout_keep_rate)
    # g_conv1 = tf.layers.batch_normalization(g_conv1)
    g_conv2 = tf.nn.conv2d_transpose(g_conv1, gw['w2'], [batch_size,  4,  4,  128], [1, 2, 2, 1], padding="SAME") + gw['b2'] # 4*4, 64 channels
    g_conv2 = activation_function(g_conv2)
    # g_conv2 = tf.nn.dropout(g_conv2, dropout_keep_rate)
    # g_conv2 = tf.layers.batch_normalization(g_conv2)
    g_conv3 = tf.nn.conv2d_transpose(g_conv2, gw['w3'], [batch_size,  7,  7,   64], [1, 2, 2, 1], padding="SAME") + gw['b3'] # 7*7, 32 channels
    g_conv3 = activation_function(g_conv3)
    # g_conv3 = tf.nn.dropout(g_conv3, dropout_keep_rate)
    # g_conv3 = tf.layers.batch_normalization(g_conv3)
    g_conv4 = tf.nn.conv2d_transpose(g_conv3, gw['w4'], [batch_size, 14, 14,  128], [1, 2, 2, 1], padding="SAME") + gw['b4'] # 14*14, 32 channels
    g_conv4 = activation_function(g_conv4)
    # g_conv4 = tf.nn.dropout(g_conv4, dropout_keep_rate)
    # g_conv4 = tf.layers.batch_normalization(g_conv4)
    g_conv5 = tf.nn.conv2d_transpose(g_conv4, gw['w5'], [batch_size, 28, 28,    1], [1, 2, 2, 1], padding="SAME") + gw['b5'] # 28*28, 1 channels
    # g_conv5 = tf.layers.batch_normalization(g_conv5)
    x = tf.nn.tanh(g_conv5)
    return x
    pass

def D(x, dw):
## this function is supposeed to discriminate the x is from 
## ground truth or generator
    d_conv1 = tf.nn.conv2d(      x, dw['w1'], [1, 2, 2, 1], 'SAME') + dw['b1'] # 14*14, 512 channels 
    d_conv1 = tf.nn.elu(d_conv1)
    d_conv2 = tf.nn.conv2d(d_conv1, dw['w2'], [1, 2, 2, 1], 'SAME') + dw['b2'] # 7*7, 128 channels 
    d_conv2 = tf.nn.elu(d_conv2)
    d_conv3 = tf.nn.conv2d(d_conv2, dw['w3'], [1, 2, 2, 1], 'SAME') + dw['b3'] # 4*4, 512 channels 
    d_conv3 = tf.nn.elu(d_conv3)
    d_conv4 = tf.nn.conv2d(d_conv3, dw['w4'], [1, 2, 2, 1], 'SAME') + dw['b4'] # 2*2, 256 channels 
    d_conv4 = tf.nn.elu(d_conv4)
    d_conv5 = tf.nn.conv2d(d_conv4, dw['w5'], [1, 1, 1, 1], 'VALID') + dw['b5'] # 1*1, 1 channels 
    return tf.reshape(d_conv5, [-1,])
    pass
    
def W(reuse=False):
    initializer = tf.contrib.layers.xavier_initializer()
    # initializer = tf.keras.initializers.he_normal()
    #initializer = tf.random_uniform
    # initializer = tf.random_normal
    with tf.variable_scope('Generator', reuse=reuse):
        gw = {
            # 'w1' : tf.Variable(initializer([ 1,  1, 256,  512])),
            'w1' : tf.Variable(tf.random_uniform([ 1,  1, 256,  512])),
            'b1' : tf.Variable(tf.zeros([256])),
            'w2' : tf.Variable(initializer([ 2,  2,  128, 256])),
            'b2' : tf.Variable(tf.zeros([128])),
            'w3' : tf.Variable(initializer([ 3,  3,   64,  128])),
            'b3' : tf.Variable(tf.zeros([64])),
            'w4' : tf.Variable(initializer([ 4,  4,  128,   64])),
            'b4' : tf.Variable(tf.zeros([128])),
            'w5' : tf.Variable(initializer([ 5,  5,    1,   128])),
            'b5' : tf.Variable(tf.zeros([1])),
        }
    # initializer = tf.random_uniform
    # initializer = tf.random_normal
    #initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope('Discriminator', reuse=reuse):
        dw = {
            'w1' : tf.Variable(initializer([3, 3, 1, 1024])),
            'b1' : tf.Variable(tf.zeros([1024])),
            'w2' : tf.Variable(initializer([3, 3, 1024, 512])),
            'b2' : tf.Variable(tf.zeros([512])),
            'w3' : tf.Variable(initializer([3, 3, 512, 256])),
            'b3' : tf.Variable(tf.zeros([256])),
            'w4' : tf.Variable(initializer([3, 3, 256, 128])),
            'b4' : tf.Variable(tf.zeros([128])),
            'w5' : tf.Variable(initializer([2, 2, 128, 1])),
            'b5' : tf.Variable(tf.zeros([1])),
        }
    return [gw,dw]
    pass 

    
def main():
    batch_size = 32
    enhance_G_sample_rate = 1 #0.25
    training_iter = 150000
    noize_dim = 512
    alpha = 1. # constant for weaking the D
    softdec_c = .05 # soft the one-hot
    mbdl_sample_no = 5 # mini batch discrimination sample number
    mbdl_lambda = 0.1 # attenuation the mini-batch discrimination penality
    
    ## Import MNIST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    
    ## build GAN
    # announce the essential placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1]) # put the real samples
    Y = tf.placeholder(dtype=tf.float32, shape=[None,]) # the real and fake labels
    Z = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, noize_dim]) # the noise for generating samples
    # announce the weights of G and D
    [gw, dw] = W()
    # Since the defalut setting of Tensorflow will update every variable during each optimizor, we need to lock the 
    # variables which we don't want it be influenced when runing optimization
    G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
    # G_var = [var for var in tf.trainable_variables() if var.name.startswith("Generator")]
    # D_var = [var for var in tf.trainable_variables() if var.name.startswith("Discriminator")]
    
    # get samples from generator
    gX = G(Z, gw)
    mbdl = mini_batch_discrimonation(mbdl_sample_no, noize_dim, gw)
    
    # define the loss and optimizers, real samples are labeled 1 and fake samples are labeled 0
    logits_4D = D(tf.concat([gX, X], axis=0), dw) #label: (fake, real)
    logits_4G = D(gX, dw)
    loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits_4D))
    loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G)) * (mbdl_lambda - mbdl) # policy gradient
    opt_D = tf.train.AdamOptimizer(1e-4, beta1=0.382).minimize(loss_D, var_list = D_var)
    opt_G = tf.train.AdamOptimizer(1e-4, beta1=0.382).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.AdamOptimizer(1e-6, beta1=0.618).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.AdamOptimizer(1e-4, beta1=0.382).minimize(loss_G)
    # opt_D = tf.train.AdamOptimizer(1e-4, beta1=0.382).minimize(loss_D)
    # opt_G = tf.train.AdamOptimizer(1e-3).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.RMSPropOptimizer(1e-4).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.RMSPropOptimizer(1e-4, momentum = 1e-8).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_G, var_list = G_var)
    # opt_G = tf.train.AdadeltaOptimizer(1e-4).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.AdadeltaOptimizer(1e-6).minimize(loss_D, var_list = D_var)
    # opt_D = tf.train.AdagradOptimizer(1e-6).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.AdagradOptimizer(1e-4).minimize(loss_G, var_list = G_var)


    ## start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        feed_in_G_sample_size = int(batch_size * enhance_G_sample_rate)
        
        # y = np.hstack([np.zeros([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real)
    
        sess.run(tf.global_variables_initializer())
        for iter in range(training_iter):
            softdec =softdec_c * np.random.random()
            y = np.hstack([np.zeros([feed_in_G_sample_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            
            x, _ = mnist.train.next_batch(batch_size)
            x = x.reshape([-1, 28, 28, 1])
            x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            
            z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            # z = np.random.uniform(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            
            ## strategy 1: update simultaneously
            # _ = sess.run(opt_D, feed_dict={X:x, Y:y, Z:z}) 
            # _ = sess.run(opt_G, feed_dict={X:x, Y:y, Z:z}) 
            # cX, Closs_D, Closs_G = sess.run([gX, loss_D, loss_G], feed_dict={X:x, Y:y, Z:z}) 
            ## strategy 2: make D stroger but update less times
            if iter % 30 == 0 : 
               for D_iter in range(30):
                   _ = sess.run(opt_D, feed_dict={X:x, Y:y, Z:z})
                   z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                   x, _ = mnist.train.next_batch(batch_size)
                   x = x.reshape([-1, 28, 28, 1])
            Closs_D, Closs_G, _ = sess.run([loss_D, loss_G, opt_G], feed_dict={X:x, Y:y, Z:z})
            ## strategy 3: update G more time (because G is hard to train)
            # for Gi in range(10):
                # x, _ = mnist.train.next_batch(batch_size)
                # x = x.reshape([-1, 28, 28, 1])
                # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
                # y = np.hstack([np.zeros([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real)
                # z = np.random.normal(size=feed_in_G_sample_size* 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                # cX, Closs_D, Closs_G, _ = sess.run([gX, loss_D, loss_G, opt_G], feed_dict={X:x, Y:y, Z:z})
            ## strategy 4: fool the D
            # if np.random.random() > .98:
                # x, _ = mnist.train.next_batch(batch_size)
                # x = x.reshape([-1, 28, 28, 1])
                # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
                # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                # y_f = np.hstack([np.ones([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real)
                # sess.run(opt_D, feed_dict={X:x, Y:y_f, Z:z}) 
            
            if iter%500 == 0 :
                print('iteration:{} loss_D:{} loss_G:{}'.format(iter, Closs_D, Closs_G))
                cX = sess.run(gX, feed_dict={X:x, Y:y, Z:z})
                for gi in range(5):
                    visg = (cX[gi].T + 1)/2
                    visr = (x[gi].T + 1)/2
                    visgp = np.vstack([visg, visg, visg]).T
                    visgp *= 255
                    visgp = sess.run(tf.image.resize_images(visgp,[64,64]))
                    visrp = np.vstack([visr, visr, visr]).T
                    visrp *= 255
                    visgp.astype(np.uint8)
                    sm.imsave('g{}.jpg'.format(gi),visgp)
                    visrp.astype(np.uint8)
                    sm.imsave('r{}.jpg'.format(gi),visrp)
            
    
    pass
    
    
    
if __name__=='__main__':
    main()
    pass

# references
# https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/