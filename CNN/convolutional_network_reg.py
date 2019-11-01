""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import
import numpy as np

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 1E-4
num_steps = 50000000
batch_size = 1
display_step = 100

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = tf.keras.layers.Conv2D(64, [3, 3], strides=[1,1], activation=tf.nn.tanh, padding='SAME')(x)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = tf.keras.layers.Conv2D(128, [3, 3], strides=[1,1], activation=tf.nn.tanh, padding='SAME')(conv1)
    for i in range(3):
        conv2 = tf.keras.layers.Conv2D(128, [3,3], strides=[1,1], activation=tf.nn.tanh, padding='SAME')(conv2)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.keras.layers.Flatten()(conv2)
    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout)
    
    fc2 = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(fc1)
    fc3 = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(fc2)
    fc4 = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(fc3)
    for i in range(25):
        fc4 = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(fc4)
    # Output, class prediction
    out = tf.keras.layers.Dense(1)(fc4)
    return out

# Construct model
logits = conv_net(X, keep_prob)
print(logits)
#exit()

# Define loss and optimizer
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#    logits=logits, labels=Y))
loss_reg_op = tf.reduce_mean(tf.abs(tf.pow( (logits - tf.cast(tf.argmax(Y,1), tf.float32)),1 )))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_reg_op)


# Evaluate model
#correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
correct_pred = tf.equal(tf.cast(tf.round(logits), tf.int32), tf.cast(tf.argmax(Y,1), tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    step = 0
    while(1):
        step += 1
    #for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: .8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            [loss, acc, c_logits] = sess.run([loss_reg_op, accuracy, tf.round(logits)], feed_dict={X: batch_x,
                                                                       Y: batch_y,
                                                                       keep_prob: 1.0})
            print("Step {} loss:{:.4f} acc:{}".format(step, loss, acc))
            
            #c_logits = sess.run(logits, feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
            print('##')
            print(c_logits)
            print(sess.run(tf.argmax(batch_y, -1)))
            #print(sess.run(tf.cast(tf.argmax(batch_y, 0), tf.int32)))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
    #                                  Y: mnist.test.labels[:256],
    #                                  keep_prob: 1.0}))
