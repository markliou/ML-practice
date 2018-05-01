""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create the neural network using tensor layers
def conv_net_tflayer(x, n_classes, dropout, reuse, is_training):
    
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = tf.layers.batch_normalization(x)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 7, activation=tf.nn.elu, padding='SAME')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        #conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv1i = tf.layers.conv2d(x, 10, 1, activation=inelu, padding='VALID')
        conv1m = tf.concat([conv1,conv1i],axis=3)
        # conv1m = tf.layers.batch_normalization(conv1m)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1m, 64, 7, activation=tf.nn.elu, padding='SAME')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        #conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        conv2i = tf.layers.conv2d(conv1m, 19, 1, activation=inelu, padding='VALID')
        # conv2 = tf.layers.batch_normalization(conv2)
        conv2m = tf.concat([conv2,conv2i,x],axis=3)

        
        # Convolution Layer with 64 filters and a kernel size of 3
        conv3 = tf.layers.conv2d(conv2m, 128, 5, activation=tf.nn.elu, padding='SAME')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        #conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        conv3i = tf.layers.conv2d(conv2m, 38, 1, activation=inelu, padding='VALID')
        # conv2 = tf.layers.batch_normalization(conv2)
        conv3m = tf.concat([conv3,conv3i,x],axis=3)
        
        # Convolution Layer with 64 filters and a kernel size of 3
        conv4 = tf.layers.conv2d(conv3m, 64, 3, activation=tf.nn.elu, padding='SAME')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        #conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        conv4i = tf.layers.conv2d(conv3m, 19, 1, activation=inelu, padding='VALID')
        # conv2 = tf.layers.batch_normalization(conv2)
        conv4m = tf.concat([conv4,conv4i,x],axis=3)
        
        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv4m)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.elu)
        fc1 = tf.layers.batch_normalization(fc1)
        fc2 = tf.layers.dense(fc1, 512, activation=tf.nn.elu)
        fc1 = tf.layers.batch_normalization(fc2)
        # Apply Dropout (if is_training is False, dropout is not applied)
        # fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc2, n_classes)
    
    return out


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def inelu(x):
    scale = 1.0
    alpha = 1.0
    return scale * tf.where(x >= 0.0, x * 0.0, alpha * tf.nn.elu(x))


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    x = tf.layers.batch_normalization(x)

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    
    return out

# # Store layers weight & bias
# weights = {
    # # 5x5 conv, 1 input, 32 outputs
    # 'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # # 5x5 conv, 32 inputs, 64 outputs
    # 'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # # fully connected, 7*7*64 inputs, 1024 outputs
    # 'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # # 1024 inputs, 10 outputs (class prediction)
    # 'out': tf.Variable(tf.random_normal([1024, num_classes]))
# }

# biases = {
    # 'bc1': tf.Variable(tf.random_normal([32])),
    # 'bc2': tf.Variable(tf.random_normal([64])),
    # 'bd1': tf.Variable(tf.random_normal([1024])),
    # 'out': tf.Variable(tf.random_normal([num_classes]))
# }

# Construct model
# logits = conv_net(X, weights, biases, keep_prob)
logits = conv_net_tflayer(X, num_classes, keep_prob, reuse=False, is_training=False)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
loss_op = tf.reduce_mean(tf.losses.hinge_loss(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=1)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
anneal_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate*0.0001)
train_op = optimizer.minimize(loss_op)
anneal_op = anneal_optimizer.minimize(loss_op * 0.2)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for annealing in range(5):
        for step in range(1, num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
        
        ##### anealing strategy
        for step in range(5):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            np.random.shuffle(batch_y)
            # sess.run(anneal_op, feed_dict={X: mnist.validation.images, Y: mnist.validation.labels, keep_prob: dropout})
            sess.run(anneal_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            
    # last optimization
    for step in range(1, num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))
