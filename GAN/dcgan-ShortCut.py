#!/usr/bin/python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.misc as sm

def mnist_select(mnist_obj, target_no, sample_no):
    cnt = 0
    target_x = [[] for i in range(sample_no)]
    while(1):
        if cnt < sample_no:
            x, y = mnist_obj.train.next_batch(1)
            if (y == target_no) or (np.random.random() > 0.95):
                target_x[cnt] = x.copy()
                cnt += 1
        else :
            return np.vstack(target_x)
        
pass

def minibatch_discrimonation(sample_no, noize_dim, gw):
    # Here will generat new samples for batch discriminator.
    # comparing to iteration run all the samples, this just generats two times and 
    # simplily compare them once to get roughtly estimation.
    # This value will be applied with focal loss which means we trend to ignore the 
    # gradients casuing the mode collapse.
    #return  1 - tf.losses.cosine_distance(
                #labels      = tf.nn.l2_normalize(G(np.random.uniform(size = sample_no * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim]), gw), 0),
                #predictions = tf.nn.l2_normalize(G(np.random.uniform(size = sample_no * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim]), gw), 0),
                #axis = 0
                #) * (1/sample_no)
    #mdl = tf.losses.absolute_difference(
    #      labels      = tf.nn.l2_normalize(G(np.random.uniform(size = sample_no * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim]), gw), 0),
    #      predictions = tf.nn.l2_normalize(G(np.random.uniform(size = sample_no * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim]), gw), 0),
    #      ) * (1/sample_no)
    
    
    GA = tf.nn.l2_normalize(G(tf.random_uniform([sample_no, 1, 1, noize_dim]), gw), 0)
    GB = tf.nn.l2_normalize(G(tf.random_uniform([sample_no, 1, 1, noize_dim]), gw), 0)
    FAc = tf.layers.conv2d(GA, 1, [4, 4], [4, 4], "VALID", trainable=False, kernel_initializer=tf.ones_initializer)
    FBc = tf.layers.conv2d(GB, 1, [4, 4], [4, 4], "VALID", trainable=False, kernel_initializer=tf.ones_initializer)
    GA = tf.nn.l2_normalize(tf.reshape(FAc,[sample_no,-1]), 0)
    GB = tf.nn.l2_normalize(tf.reshape(FBc,[sample_no,-1]), 0)
    
    mdl = tf.losses.mean_squared_error(
          labels      = GA,
          predictions = GB,
          ) * (1/sample_no)
          
    # mdl = (tf.losses.cosine_distance(
            # labels      = GA , 
            # predictions = GB , 
            # axis = 0
            # ) * (1/sample_no))
    # mdl -= 0.5
    return mdl
pass

def minibatch_discrimonation_ind(sample_no, noize_dim, gw, gX):
    # this section try to return the focal coefficents of each generated sample.
    k = 2 # scale constant
    gXc1 = tf.layers.conv2d(gX, 1, [4, 4], [4, 4], "VALID", trainable=False, kernel_initializer=tf.ones_initializer, activation=None)
    
    mdl = tf.zeros([tf.shape(gX)[0]])
    for i in range(sample_no):
        # gS = tf.nn.l2_normalize( G(tf.random_uniform([tf.shape(gX)[0], 1, 1, noize_dim]), gw), 0)
        gS = G(tf.random_uniform([tf.shape(gX)[0], 1, 1, noize_dim]), gw)
        gSc1 = tf.layers.conv2d(gS, 1, [4, 4], [4, 4], "VALID", trainable=False, kernel_initializer=tf.ones_initializer, activation=None)
        
        # mdl +=  tf.reduce_mean(
                    # #tf.reshape(
                        # tf.losses.mean_squared_error(
                            # labels      = tf.nn.l2_normalize(tf.reshape(gSc1,[tf.shape(gX)[0],-1]), 0),
                            # predictions = tf.nn.l2_normalize(tf.reshape(gXc1,[tf.shape(gX)[0],-1]), 0),
                            # reduction = tf.losses.Reduction.NONE
                        # ),
                    # #    [tf.shape(gX)[0], -1],
                    # #),
                    # axis = -1
                # )
                
        mdl +=  tf.reduce_mean(
                    #tf.reshape(
                        tf.losses.cosine_distance(
                            labels      = tf.nn.l2_normalize(tf.reshape(gSc1,[tf.shape(gX)[0],-1]), 0),
                            predictions = tf.nn.l2_normalize(tf.reshape(gXc1,[tf.shape(gX)[0],-1]), 0),
                            reduction = tf.losses.Reduction.NONE,
                            axis = -1
                            # dim = -1
                        ),
                    #    [tf.shape(gX)[0], -1],
                    #),
                    axis = -1
                )
                
    mdl /= sample_no
    
    # reward normalization
    mdl_max, mdl_min = tf.reduce_max(mdl), tf.reduce_min(mdl)  
    mdl = (mdl - mdl_min)/(mdl_max - mdl_min)
    # mdl = tf.nn.sigmoid(mdl)
    mdl = (mdl - 0.5) * k
    
    return mdl 
pass

def feature_map_upsampling(X, size, method=tf.image.ResizeMethod.BICUBIC):
    PX = X * 1e10
    PX = tf.cast(PX, tf.int32)
    PX = tf.image.resize_images(PX, size, method)
    PX = tf.cast(PX, tf.float32)
    PX /= 1e10
    return PX
pass

# def G(Z, gw, trainable=True, reuse=tf.AUTO_REUSE):
def G(Z, gw, trainable=True, reuse=True, training=True):
## this function is setted to generate a 28*28 digit picture
## from a normal distribution z
## here, z with 32 channels is setted to size of 7*7 which can be transpose convoluted 
## 2 times with stride 2 to get 28*28 matrix

    activation_function = tf.nn.elu
    # activation_function = tf.nn.leaky_relu
    # activation_function = tf.nn.tanh
    dropout_keep_rate = .5
    # dropout_keep_rate = tf.random_uniform([], minval=.5, maxval=1.)
    
    with tf.device('/device:GPU:0'):
        with tf.variable_scope('Generator', reuse=reuse):
            Z = tf.convert_to_tensor(Z)
            Z = tf.cast(Z, tf.float32)
            # Z = tf.nn.tanh(Z)
            
            # the shape of z should be [-1, 1, 1, 128]
            batch_size = tf.shape(Z)[0]
        
            # Z = tf.nn.tanh(Z)
            # Z = tf.layers.batch_normalization(Z, epsilon=1e-5, training=training)
            g_conv1 = tf.nn.conv2d_transpose(      Z, gw['w1'], [batch_size,  2,  2,  512], [1, 2, 2, 1], padding="SAME") + gw['b1'] # 2*2, 128 channels
            g_conv1 = activation_function(g_conv1)
            # g_conv1 = tf.add(g_conv1, tf.layers.conv2d(feature_map_upsampling(Z, [2, 2], tf.image.ResizeMethod.NEAREST_NEIGHBOR), 1, [3, 3], [1, 1], "SAME", trainable=False, activation=None, kernel_initializer=tf.constant_initializer(0.1)))/2.
            g_conv1 = tf.layers.batch_normalization(g_conv1, epsilon=1e-5, training=training)
            g_conv1 = tf.layers.conv2d(g_conv1, 512, [1, 1], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(1, 1))
            # g_conv1 = tf.layers.batch_normalization(g_conv1, epsilon=1e-5, training=training)
            # g_conv1 = tf.layers.conv2d(g_conv1, 512, [1, 1], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(2, 2))
            # g_conv1 = tf.layers.batch_normalization(g_conv1, epsilon=1e-5, training=training)
            # g_conv1 = tf.layers.conv2d(g_conv1, 512, [1, 1], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(1, 1))
            # g_conv1 = tf.layers.batch_normalization(g_conv1, epsilon=1e-5, training=training)
            # g_conv1 = tf.layers.conv2d(g_conv1, 512, [1, 1], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv1 = tf.nn.dropout(g_conv1, dropout_keep_rate)
            g_conv1 = tf.concat([g_conv1, feature_map_upsampling(Z, [2, 2], tf.image.ResizeMethod.NEAREST_NEIGHBOR)], 3)
            # g_conv1 = tf.layers.batch_normalization(g_conv1, epsilon=1e-5, training=training)
            g_conv1 = tf.layers.conv2d(g_conv1, 512, [1, 1], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv1 = tf.layers.batch_normalization(g_conv1, epsilon=1e-5, training=training)
            g_conv2 = tf.nn.conv2d_transpose(g_conv1, gw['w2'], [batch_size,  4,  4,  256], [1, 2, 2, 1], padding="SAME") + gw['b2'] # 4*4, 64 channels
            g_conv2 = activation_function(g_conv2)
            g_conv2 = tf.layers.batch_normalization(g_conv2, epsilon=1e-5, training=training)
            # g_conv2 = tf.add(g_conv2, tf.layers.conv2d(feature_map_upsampling(g_conv1, [4, 4], tf.image.ResizeMethod.NEAREST_NEIGHBOR), 1, [3, 3], [1, 1], "SAME", trainable=False, activation=None, kernel_initializer=tf.constant_initializer(0.1)))/2.
            g_conv2 = tf.layers.batch_normalization(g_conv2, epsilon=1e-5, training=training)
            g_conv2 = tf.layers.conv2d(g_conv2, 256, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(1, 1))
            g_conv2 = tf.layers.batch_normalization(g_conv2, epsilon=1e-5, training=training)
            g_conv2 = tf.layers.conv2d(g_conv2, 256, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(2, 2))
            g_conv2 = tf.layers.batch_normalization(g_conv2, epsilon=1e-5, training=training)
            g_conv2 = tf.layers.conv2d(g_conv2, 256, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(1, 1))
            # g_conv2 = tf.layers.batch_normalization(g_conv2, epsilon=1e-5, training=training)
            # g_conv2 = tf.layers.conv2d(g_conv2, 256, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(3, 3))
            # g_conv2 = tf.layers.batch_normalization(g_conv2, epsilon=1e-5, training=training)
            # g_conv2 = tf.layers.conv2d(g_conv2, 256, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(1, 1))
            g_conv2 = tf.nn.dropout(g_conv2, dropout_keep_rate)
            g_conv2 = tf.concat([g_conv2, feature_map_upsampling(g_conv1, [4, 4], tf.image.ResizeMethod.NEAREST_NEIGHBOR)], 3)
            g_conv2 = tf.layers.batch_normalization(g_conv2, epsilon=1e-5, training=training)
            g_conv2 = tf.layers.conv2d(g_conv2, 256, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(1, 1))
            # g_conv2 = tf.layers.batch_normalization(g_conv2, epsilon=1e-5, training=training)
            # g_conv2 = tf.layers.conv2d(g_conv2, 512, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv2 = tf.layers.batch_normalization(g_conv2, epsilon=1e-5, training=training, renorm=True)
            g_conv3 = tf.nn.conv2d_transpose(g_conv2, gw['w3'], [batch_size,  7,  7, 128], [1, 2, 2, 1], padding="SAME") + gw['b3'] # 7*7, 32 channels
            g_conv3 = activation_function(g_conv3)
            # g_conv3 = tf.add(g_conv3, tf.layers.conv2d(feature_map_upsampling(g_conv2, [7, 7], tf.image.ResizeMethod.NEAREST_NEIGHBOR), 1, [3, 3], [1, 1], "SAME", trainable=False, activation=None, kernel_initializer=tf.constant_initializer(0.1)))/2.
            g_conv3 = tf.layers.batch_normalization(g_conv3, epsilon=1e-5, training=training)
            g_conv3 = tf.layers.conv2d(g_conv3, 128, [5, 5], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv3 = tf.layers.batch_normalization(g_conv3, epsilon=1e-5, training=training)
            g_conv3 = tf.layers.conv2d(g_conv3, 128, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(2, 2))
            g_conv3 = tf.layers.batch_normalization(g_conv3, epsilon=1e-5, training=training)
            g_conv3 = tf.layers.conv2d(g_conv3, 128, [5, 5], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            # g_conv3 = tf.layers.batch_normalization(g_conv3, epsilon=1e-5, training=training)
            # g_conv3 = tf.layers.conv2d(g_conv3, 128, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(3, 3))
            # g_conv3 = tf.layers.batch_normalization(g_conv3, epsilon=1e-5, training=training)
            # g_conv3 = tf.layers.conv2d(g_conv3, 128, [5, 5], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv3 = tf.nn.dropout(g_conv3, dropout_keep_rate)
            g_conv3 = tf.concat([g_conv3, feature_map_upsampling(g_conv2, [7, 7], tf.image.ResizeMethod.NEAREST_NEIGHBOR)], 3)
            # g_conv3 = tf.layers.batch_normalization(g_conv3, epsilon=1e-5, training=training)
            g_conv3 = tf.layers.conv2d(g_conv3, 128, [5, 5], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            # g_conv3 = tf.layers.batch_normalization(g_conv3, epsilon=1e-5, training=training)
            # g_conv3 = tf.layers.conv2d(g_conv3, 256, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv3 = tf.layers.batch_normalization(g_conv3, epsilon=1e-5, training=training, renorm=True)
            g_conv4 = tf.nn.conv2d_transpose(g_conv3, gw['w4'], [batch_size, 14, 14,   64], [1, 2, 2, 1], padding="SAME") + gw['b4'] # 14*14, 32 channels
            g_conv4 = activation_function(g_conv4)
            # g_conv4 = tf.add(g_conv4, tf.layers.conv2d(feature_map_upsampling(g_conv3, [14, 14], tf.image.ResizeMethod.NEAREST_NEIGHBOR), 1, [3, 3], [1, 1], "SAME", trainable=False, activation=None, kernel_initializer=tf.constant_initializer(0.1)))/2.
            g_conv4 = tf.layers.batch_normalization(g_conv4, epsilon=1e-5, training=training)
            g_conv4 = tf.layers.conv2d(g_conv4, 64, [7, 7], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv4 = tf.layers.batch_normalization(g_conv4, epsilon=1e-5, training=training)
            g_conv4 = tf.layers.conv2d(g_conv4, 64, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(2, 2))
            g_conv4 = tf.layers.batch_normalization(g_conv4, epsilon=1e-5, training=training)
            g_conv4 = tf.layers.conv2d(g_conv4, 64, [7, 7], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            # g_conv4 = tf.layers.batch_normalization(g_conv4, epsilon=1e-5, training=training)
            # g_conv4 = tf.layers.conv2d(g_conv4, 64, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer(), dilation_rate=(3, 3))
            # g_conv4 = tf.layers.batch_normalization(g_conv4, epsilon=1e-5, training=training)
            # g_conv4 = tf.layers.conv2d(g_conv4, 64, [7, 7], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv4 = tf.nn.dropout(g_conv4, dropout_keep_rate)
            g_conv4 = tf.concat([g_conv4, feature_map_upsampling(g_conv3, [14, 14], tf.image.ResizeMethod.NEAREST_NEIGHBOR)], 3)
            # g_conv4 = tf.layers.batch_normalization(g_conv4, epsilon=1e-5, training=training)
            g_conv4 = tf.layers.conv2d(g_conv4, 64, [7, 7], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv4 = tf.layers.batch_normalization(g_conv4, epsilon=1e-5, training=training, renorm=True)
            g_conv5 = tf.nn.conv2d_transpose(g_conv4, gw['w5'], [batch_size, 28, 28,    1], [1, 2, 2, 1], padding="SAME") + gw['b5'] # 28*28, 1 channels
            g_conv5 = tf.concat([g_conv5, feature_map_upsampling(g_conv4, [28, 28], tf.image.ResizeMethod.NEAREST_NEIGHBOR)], 3)
            g_conv5 = activation_function(g_conv5)
            g_conv5 = tf.layers.batch_normalization(g_conv5, epsilon=1e-5, training=training)
            g_conv5 = tf.layers.conv2d(g_conv5, 32, [9, 9], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv5 = tf.layers.batch_normalization(g_conv5, epsilon=1e-5, training=training)
            g_conv5 = tf.layers.conv2d(g_conv5,  1, [9, 9], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            g_conv5 = tf.layers.batch_normalization(g_conv5, epsilon=1e-5, training=training, renorm=True)
            x = tf.nn.tanh(g_conv5)
    return x
pass

# def D(x, dw, trainable=True, reuse=tf.AUTO_REUSE ):
def D(x, dw, trainable=True, reuse=True):
## this function is supposeed to discriminate the x is from 
## ground truth or generator
    dropout_keep_rate = .5 
    activation_function = tf.nn.relu
    
    with tf.device('/device:GPU:0'):
        with tf.variable_scope('Discriminator', reuse=reuse):
            d_conv1 = tf.nn.conv2d(      x, dw['w1'], [1, 2, 2, 1],  'SAME') + dw['b1'] # 14*14, 512 channels 
            d_conv1 = tf.nn.relu(d_conv1)
            # d_conv1 = tf.layers.conv2d(d_conv1, 1024, [3, 3], [1, 1],"SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            d_conv1 = tf.nn.dropout(d_conv1, dropout_keep_rate)
            d_conv2 = tf.nn.conv2d(d_conv1, dw['w2'], [1, 2, 2, 1],  'SAME') + dw['b2'] # 7*7, 128 channels 
            d_conv2 = tf.nn.relu(d_conv2)
            # d_conv2 = tf.layers.conv2d(d_conv2, 512, [3, 3], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            d_conv2 = tf.nn.dropout(d_conv2, dropout_keep_rate)
            d_conv3 = tf.nn.conv2d(d_conv2, dw['w3'], [1, 2, 2, 1],  'SAME') + dw['b3'] # 4*4, 512 channels 
            d_conv3 = tf.nn.relu(d_conv3)
            # d_conv3 = tf.layers.conv2d(d_conv3, 1024, [3, 3], [1, 1],"SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            d_conv3 = tf.nn.dropout(d_conv3, dropout_keep_rate)
            d_conv4 = tf.nn.conv2d(d_conv3, dw['w4'], [1, 2, 2, 1],  'SAME') + dw['b4'] # 2*2, 256 channels 
            d_conv4 = tf.nn.relu(d_conv4)
            # d_conv4 = tf.layers.conv2d(d_conv4,  512, [1, 1], [1, 1], "SAME", trainable=trainable, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
            d_conv4 = tf.nn.dropout(d_conv4, dropout_keep_rate)
            d_conv5 = tf.nn.conv2d(d_conv4, dw['w5'], [1, 1, 1, 1], 'VALID') + dw['b5'] # 1*1, 1 channels 
            # d_conv5 = tf.nn.relu(d_conv5)
            # d_conv5 = tf.layers.conv2d(d_conv5, 1, [1, 1], [1, 1], "VALID", trainable=True, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return tf.reshape(d_conv5, [-1,])
pass
    
def W(reuse=True):
    initializer = tf.contrib.layers.xavier_initializer()
    # initializer = tf.keras.initializers.he_normal()
    # initializer = tf.random_uniform
    # initializer = tf.random_normal
    
    with tf.device('/device:GPU:0'):
        with tf.variable_scope('Generator', reuse=reuse):
            gw = {
                'w1' : tf.Variable(initializer([ 1,  1,  512, 1024])),
                'b1' : tf.Variable(tf.zeros([512])),
                'w2' : tf.Variable(initializer([ 2,  2,  256, 512])),
                'b2' : tf.Variable(tf.zeros([256])),
                'w3' : tf.Variable(initializer([ 3,  3,  128, 256])),
                'b3' : tf.Variable(tf.zeros([128])),
                'w4' : tf.Variable(initializer([ 5,  5,   64, 128])),
                'b4' : tf.Variable(tf.zeros([64])),
                'w5' : tf.Variable(initializer([ 7,  7,    1,  64])),
                'b5' : tf.Variable(tf.zeros([1])),
            }
    # initializer = tf.random_uniform
    # initializer = tf.random_normal
    #initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('Discriminator', reuse=reuse):
            dw = {
                'w1' : tf.Variable(initializer([3, 3,    1,   512])),
                'b1' : tf.Variable(tf.zeros([512])),
                'w2' : tf.Variable(initializer([3, 3,  512,   256])),
                'b2' : tf.Variable(tf.zeros([256])),
                'w3' : tf.Variable(initializer([3, 3,  256,  512])),
                'b3' : tf.Variable(tf.zeros([512])),
                'w4' : tf.Variable(initializer([3, 3,  512,  256])),
                'b4' : tf.Variable(tf.zeros([256])),
                'w5' : tf.Variable(initializer([2, 2,  256,   1])),
                'b5' : tf.Variable(tf.zeros([1])),
            }
    return [gw,dw]
pass 

    
def main():
    batch_size = 32
    enhance_G_sample_rate = 8/32
    output_g_sample_size = 5
    training_iter = 500000
    noize_dim = 1024
    alpha = 1. # constant for weaking the D
    softdec_c = 1e-1 # soft the one-hot
    # softdec_c = .0 # turn off soft the one-hot
    mbdl_sample_no = int( batch_size * (enhance_G_sample_rate + 1) ) * 3 # mini batch discrimination sample number
    mbdli_no = 20
    
    ## Import MNIST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    
    
    ## build GAN
    # announce the essential placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1]) # put the real samples
    Y = tf.placeholder(dtype=tf.float32, shape=[None,]) # the real and fake labels
    Z = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, noize_dim]) # the noise for generating samples
    # announce the weights of G and D
    [gw, dw] = W()
    
    # get samples from generator
    gX  = G(Z, gw, reuse=False)
    gXd = G(Z, gw, reuse=True, training=False)
    mbdl = minibatch_discrimonation(mbdl_sample_no, noize_dim, gw)
    mbdliG = minibatch_discrimonation_ind(mbdli_no, noize_dim, gw, gX)
    mbdliD = minibatch_discrimonation_ind(mbdli_no, noize_dim, gw, tf.concat([gX, X], axis=0))
    # mbdliDr = minibatch_discrimonation_ind(30, noize_dim, gw, X)
    # mbdliDf = minibatch_discrimonation_ind(30, noize_dim, gw, gX)
    # mbdliD = tf.concat([mbdliDf, mbdliDr], axis=0)
    # mbdliD = tf.concat([mbdliDf, tf.ones_like(mbdliDr)], axis=0)
    
    # define the loss and optimizers, real samples are labeled 1 and fake samples are labeled 0
    logits_4Df = D(gX, dw, reuse=False) #label: (fake, real)
    logits_4Dr = D( X, dw)
    logits_4D  = tf.concat([logits_4Df, logits_4Dr], axis=0)
    logits_4G = D(gX, dw)
    
    ## loss definition
    # focal loss, using the minibatch distrinimation result as focal. The weighs with large resutls are bad and will get large gradients.
    # loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits_4D) * tf.pow(tf.exp(-mbdl), 1))  
    # loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits_4D) * tf.pow(tf.exp(-mbdliD), 2))
    loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits_4D) )
    # loss_D = tf.reduce_mean(tf.losses.hinge_loss(labels=Y, logits=logits_4D, reduction=tf.losses.Reduction.NONE) )
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G) * tf.pow(tf.exp(mbdl), 1)) 
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G) * tf.pow(tf.exp(mbdliG), 2))
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G) )
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G) )
    loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4Df), logits=logits_4Df) )
    # loss_G = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_4Df), logits=logits_4Df) )
    # loss_G = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_4G), logits=logits_4G) )
    # loss_G = -tf.reduce_min(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_4G), logits=logits_4G) )
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_4G), logits=logits_4G) )#https://github.com/soumith/ganhacks # 2-flip label
    # regulize the loss 
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G)) + 0.01 * mbdl
    
    ## optimizer definition
    
    # Since the defalut setting of Tensorflow will update every variable during each optimizor, we need to lock the 
    # variables which we don't want it be influenced when runing optimization
    G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
    # G_var = [var for var in tf.trainable_variables() if var.name.startswith("Generator")]
    # D_var = [var for var in tf.trainable_variables() if var.name.startswith("Discriminator")]
    
    
    # opt_D = tf.train.AdamOptimizer(1e-5, beta1=0.618).minimize(loss_D)
    # opt_G = tf.train.AdamOptimizer(1e-5, beta1=0.618).minimize(loss_G)
    # opt_D = tf.train.AdamOptimizer(1e-3).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.AdamOptimizer(1e-3).minimize(loss_G, var_list = G_var)# opt_D = tf.train.AdamOptimizer(1e-5, beta1=0.382).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.AdamOptimizer(1e-4, beta1=0.618).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.AdamOptimizer(1e-4, beta1=0.618).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.AdamOptimizer(1e-4, beta1=0.618).minimize(loss_G)
    # opt_D = tf.train.AdamOptimizer(1e-5, beta1=0.382).minimize(loss_D, var_list = D_var)
    opt_G = tf.train.AdamOptimizer(1e-5, beta1=0.618).minimize(loss_G, var_list = G_var) 
    # opt_D = tf.train.AdamOptimizer(1e-4, beta1=0.382).minimize(loss_D)
    # opt_G = tf.train.AdamOptimizer(1e-3).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.RMSPropOptimizer(1e-5, momentum=0.382).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.RMSPropOptimizer(1e-5, momentum=0.382).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_G, var_list = G_var)
    # opt_G = tf.train.AdadeltaOptimizer(1e-4).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.AdadeltaOptimizer(1e-6).minimize(loss_D, var_list = D_var)
    # opt_D = tf.train.AdagradOptimizer(1e-4).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.AdagradOptimizer(1e-4).minimize(loss_G, var_list = G_var)
    # opt_D = tf.contrib.opt.NadamOptimizer(1e-5, beta1=0.382).minimize(loss_D, var_list = D_var)
    # opt_G = tf.contrib.opt.NadamOptimizer(1e-5, beta1=0.382).minimize(loss_G, var_list = G_var)
    opt_D = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-8, learning_rate=1e-5, beta1=0.618).minimize(loss_D, var_list = D_var)
    # opt_G = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-8, learning_rate=1e-5, beta1=0.618).minimize(loss_G, var_list = G_var)
    

    ## start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    with tf.Session(config=config) as sess:
        feed_in_G_sample_size = int(batch_size * enhance_G_sample_rate)
        
        # y = np.hstack([np.zeros([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real)
    
        sess.run(tf.global_variables_initializer())
        for iter in range(training_iter):
            # softdec = softdec_c * np.random.random()
            # softdec = softdec_c # without noise
            
            # y = np.hstack([np.zeros([feed_in_G_sample_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            
            # x, _ = mnist.train.next_batch(batch_size)
            # # x = mnist_select(mnist, int(iter/300) % 9, batch_size) # this will enforce the number class in to the dataset
            # x = x.reshape([-1, 28, 28, 1])
            # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            
            # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            # # z = np.random.uniform(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            
            ## strategy 1: update simultaneously
            #####
            ## D should be trained using equal number of fake and real samples.
            ## More training samples are also feeded into D than G when training.
            ## During optimize G, if G generate less samples, the gradients would be more specific to G, 
            ## and this make G can understand more about which parts should be modified.
            ## So the strategy here uses few generated sample, but more real sample to D for getting the gradients.
            #####
            # # train D
            # x, _ = mnist.train.next_batch(batch_size)
            # x = x.reshape([-1, 28, 28, 1])
            # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            # # y = np.hstack([np.zeros([batch_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            # y = np.hstack([np.zeros([batch_size,]), np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            # #z = np.random.normal(size = batch_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            # # since 0~9 should be sampled equally, the final channel are set the uniform rather than normaldistribution
            # z = np.random.normal(size = batch_size * 1 * 1 * (noize_dim-1)).reshape([-1, 1, 1, noize_dim-1])
            # z = np.concatenate((z, np.random.uniform(size=[batch_size, 1, 1, 1]) * 2 - 1), axis=-1)
            # # z = np.concatenate((z, np.random.randint(10, size=[batch_size, 1, 1, 1]) * .2 - 1), axis=-1)
            
            
            # # _, CmbdliD = sess.run([opt_D, mbdliD], feed_dict={X:x, Y:y, Z:z}) 
            # _ = sess.run([opt_D], feed_dict={X:x, Y:y, Z:z})
            
            # # train G
            # x, _ = mnist.train.next_batch(batch_size)
            # x = x.reshape([-1, 28, 28, 1])
            # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            # # y = np.hstack([np.zeros([feed_in_G_sample_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            # y = np.hstack([np.zeros([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real), soft one-hots
            # # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * (noize_dim-1)).reshape([-1, 1, 1, noize_dim-1])
            # z = np.concatenate((z, np.random.uniform(size=[feed_in_G_sample_size, 1, 1, 1]) * 2 - 1), axis=-1)
            # # z = np.concatenate((z, np.random.randint(10, size=[feed_in_G_sample_size, 1, 1, 1]) * .2 - 1), axis=-1)
            # # _, CmbdliG = sess.run([opt_G, mbdliG], feed_dict={X:x, Y:y, Z:z}) 
            # _ = sess.run([opt_G], feed_dict={Z:z})
            
            # # get the output
            # cX, Closs_D, Closs_G, Cmbdl = sess.run([gX, loss_D, loss_G, mbdl], feed_dict={X:x, Y:y, Z:z}) 
            
            ## strategy 2: make D stroger but update less times
            # # train D
            # if iter % 100 == 0 : 
                # for D_iter in range(20):
                    # x, _ = mnist.train.next_batch(batch_size)
                    # x = x.reshape([-1, 28, 28, 1])
                    # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
                    # z = np.random.normal(size = batch_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                    # y = np.hstack([np.zeros([batch_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
                    # # _, CmbdliD = sess.run([opt_D, mbdliD], feed_dict={X:x, Y:y, Z:z}) 
                    # _ = sess.run([opt_D], feed_dict={X:x, Y:y, Z:z})
            
            # # train G
            # x, _ = mnist.train.next_batch(batch_size)
            # x = x.reshape([-1, 28, 28, 1])
            # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            # y = np.hstack([np.zeros([feed_in_G_sample_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            # # _, CmbdliG = sess.run([opt_G, mbdliG], feed_dict={Z:z}) 
            # _ = sess.run([opt_G], feed_dict={Z:z}) 
            
            # # get the output
            # # cX, Closs_D, Closs_G, Cmbdl = sess.run([gX, loss_D, loss_G, mbdl], feed_dict={X:x, Y:y, Z:z}) 
            # cX, Closs_D, Closs_G = sess.run([gX, loss_D, loss_G], feed_dict={X:x, Y:y, Z:z}) 
            
            ## strategy 3: update G more time (because G is hard to train) and fix D
            # train D
            for Di in range(1):
                softdec = softdec_c * np.random.random()
                generated_sample_size_for_D = int(batch_size * 1.)
                x, _ = mnist.train.next_batch(batch_size)
                x = x.reshape([-1, 28, 28, 1])
                x = (x - 0.5)/ 0.5 # normalize from -1 to 1
                # y = np.hstack([np.zeros([batch_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
                y = np.hstack([np.zeros([generated_sample_size_for_D,]), np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
                z = np.random.normal(size = generated_sample_size_for_D * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                # since 0~9 should be sampled equally, the final channel are set the uniform rather than normaldistribution
                # z = np.random.normal(size = generated_sample_size_for_D * 1 * 1 * (noize_dim-1)).reshape([-1, 1, 1, noize_dim-1])
                # z = np.concatenate((z, np.random.uniform(size=[generated_sample_size_for_D, 1, 1, 1]) * 2 - 1), axis=-1)
                # z = np.concatenate((z, np.random.randint(10, size=[batch_size, 1, 1, 1]) * .2 - 1), axis=-1)
                
                # _, CmbdliD = sess.run([opt_D, mbdliD], feed_dict={X:x, Y:y, Z:z}) 
                _ = sess.run([opt_D], feed_dict={X:x, Y:y, Z:z})
            
            # train G
            x, _ = mnist.train.next_batch(batch_size)
            x = x.reshape([-1, 28, 28, 1])
            x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            # y = np.hstack([np.zeros([feed_in_G_sample_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            y = np.hstack([np.zeros([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real), soft one-hots
            for Gi in range(5):
                z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * (noize_dim-1)).reshape([-1, 1, 1, noize_dim-1])
                # z = np.concatenate((z, np.random.uniform(size=[feed_in_G_sample_size, 1, 1, 1]) * 2 - 1), axis=-1)
                # z = np.concatenate((z, np.random.randint(10, size=[feed_in_G_sample_size, 1, 1, 1]) * .2 - 1), axis=-1)
                # _, CmbdliG = sess.run([opt_G, mbdliG], feed_dict={X:x, Y:y, Z:z}) 
                _ = sess.run([opt_G], feed_dict={Z:z})
            
            # get the output
            # cX, Closs_D, Closs_G, Cmbdl = sess.run([gX, loss_D, loss_G, mbdl], feed_dict={X:x, Y:y, Z:z}) 
            Closs_D, Closs_G, _ = sess.run([loss_D, loss_G, opt_G], feed_dict={X:x, Y:y, Z:z})
            
            ## strategy 4: fool the D
            # if np.random.random() > .98:
                # x, _ = mnist.train.next_batch(batch_size)
                # x = x.reshape([-1, 28, 28, 1])
                # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
                # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                # y_f = np.hstack([np.ones([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real)
                # sess.run(opt_D, feed_dict={X:x, Y:y_f, Z:z}) 
            
            if iter % 500 == 0 :
                # print('iteration:{} loss_D:{} loss_G:{} mbdliD:{} mbdliG:{} mbdl:{}'.format(iter, Closs_D, Closs_G, CmbdliD, CmbdliG, Cmbdl))
                print('iteration:{} loss_D:{} loss_G:{}'.format(iter, Closs_D, Closs_G))
                    
                for gi in range(output_g_sample_size):
                    z = np.random.normal(size = output_g_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                    # z = np.random.normal(size = 1 * 1 * 1 * (noize_dim-1)).reshape([-1, 1, 1, noize_dim-1])
                    # z = np.concatenate((z, np.random.uniform(size=[1, 1, 1, 1]) * 2 - 1), axis=-1)
                    # z = np.concatenate((z, np.random.randint(10, size=[1, 1, 1, 1]) * .2 - 1), axis=-1)
                    cX = sess.run(gX, feed_dict={Z:z})
                    visg = (cX[0].T + 1)/2
                    # visr = (x[0].T + 1)/2
                    visgp = np.vstack([visg, visg, visg]).T
                    visgp *= 255
                    visgp = sess.run(tf.image.resize_images(visgp,[64,64]))
                    # visrp = np.vstack([visr, visr, visr]).T
                    # visrp *= 255
                    visgp.astype(np.uint8)
                    sm.imsave('G/g{}_{}.jpg'.format(gi, iter),visgp)
                    # visrp.astype(np.uint8)
                    # sm.imsave('G/r{}_{}.jpg'.format(gi, iter),visrp)
            
    
pass
    
    
    
if __name__=='__main__':
    main()
pass

# references
# https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
# https://adoberesearch.ctlprojects.com/wp-content/uploads/2018/04/CVPR2018_Paper3623_Chang.pdf
# https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9
