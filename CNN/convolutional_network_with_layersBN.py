'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist = input_data.read_data_sets("./", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 250
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
#keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    #return tf.nn.relu(x)
    #return tf.nn.tanh(x)
    #return tf.nn.elu(x)
    return selu(x)

# selu in tensorflow   markliou
def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
    
# batch normalization layer markliou
def BN(x, beta=0, gamma=1, epsilon=0.1):
    axis = list(range(len(x.get_shape()) - 1))
    mean, variance = tf.nn.moments(x, axis)
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-3)
pass

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    x = tf.layers.batch_normalization(x, training=True, trainable=True)
    

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    # conv1 = maxpool2d(conv1, k=2)
    #x = BN(x) # batch normalization  markliou
    conv1 = tf.layers.batch_normalization(conv1, training=True, trainable=True)
    
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    # conv2 = maxpool2d(conv2, k=2)
    #conv2 = BN(conv2) # batch normalization  markliou
    conv2 = tf.layers.batch_normalization(conv2, training=True, trainable=True)
    
    # appending layers     markliou
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # conv3 = BN(conv3) # batch normalization  markliou
    conv3 = tf.layers.batch_normalization(conv3, training=True, trainable=True)
    
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # conv4 = BN(conv4) # batch normalization  markliou
    # conv4 = tf.layers.batch_normalization(conv4, training=True, trainable=True)
    
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # conv5 = BN(conv5) # batch normalization  markliou
    conv5 = tf.layers.batch_normalization(conv5, training=True, trainable=True)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    #fc1 = tf.nn.relu(fc1)
    fc1 = selu(fc1)
    
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = selu(fc2)
    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = selu(fc3)
    
    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 128 outputs
    'wc1': tf.Variable(tf.random_normal([7, 7, 1, 128]   ) ),
    # 5x5 conv, 128 inputs, 512 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 128, 128] ) ),
    
    # 5x5 conv, 512 inputs, 512 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 512] ) ),
    # 5x5 conv, 512 inputs, 128 outputs
    'wc4': tf.Variable(tf.random_normal([1, 1, 512, 128] ) ),
    # 5x5 conv, 128 inputs, 64 outputs
    'wc5': tf.Variable(tf.random_normal([1, 1, 128, 64]  ) ),
    
    # fully connected, 7*7*64 inputs, 1024 outputs
    #'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'wd1': tf.Variable(tf.random_normal([28*28*64, 1024])),
    
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'wd3': tf.Variable(tf.random_normal([1024, 1024])),
    
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([128])),
    'bc2': tf.Variable(tf.random_normal([128])),
    
    'bc3': tf.Variable(tf.random_normal([512])),
    'bc4': tf.Variable(tf.random_normal([128])),
    'bc5': tf.Variable(tf.random_normal([64])),
    
    'bd1': tf.Variable(tf.random_normal([1024])),
    
    'bd2': tf.Variable(tf.random_normal([1024])),
    'bd3': tf.Variable(tf.random_normal([1024])),
    
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
# pred = conv_net(x, weights, biases, keep_prob)
pred = conv_net(x, weights, biases, 0)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
    #while 1:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        # sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       # keep_prob: dropout})
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        
        # print(batch_y[12])
        # print(len(batch_x[12]))
        # for i in range(0,28,1):
        #     print(batch_x[12][i*28:i*28+28])
        # exit()

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            # loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              # y: batch_y,
                                                              # keep_prob: 1.})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        # sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      # y: mnist.test.labels[:256],
                                      # keep_prob: 1.}))
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256]}))
