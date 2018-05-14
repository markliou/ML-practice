""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


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


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    # conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    #conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    #conv2 = maxpool2d(conv2, k=2)

    ## for fully-convolutional NN
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides=2) #14
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], strides=2) #7
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], strides=2) #4
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], strides=2) #2
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], strides=2) #1

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    # fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # fc1 = tf.nn.relu(fc1)
    # # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)
    
    # Output, class prediction
    # out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    
    out = tf.nn.conv2d(conv5, weights['out'], strides=[1,1,1,1],padding='SAME') + biases['out'] #1
    
    out = tf.reshape(out, [-1,num_classes])
    
    return out

# Store layers weight & bias
# initializer = tf.contrib.layers.xavier_initializer()
initializer = tf.random_normal()
weights = {
    
    # Xavior initializer
    'wc1': tf.Variable(initializer([3, 3, 1, 32])),
    'wc2': tf.Variable(initializer([3, 3, 32, 64])),
    'wc3': tf.Variable(initializer([3, 3, 64, 32])),
    'wc4': tf.Variable(initializer([3, 3, 32, 64])),
    'wc5': tf.Variable(initializer([3, 3, 64, 32])),
    
    # 'out': tf.Variable(tf.random_normal([3, 3, 32, num_classes]))
    'out': tf.Variable(initializer([1, 1, 32, num_classes]))
}

biases = {
    'bc1': tf.Variable(initializer([32])),
    'bc2': tf.Variable(initializer([64])),
    'bc3': tf.Variable(initializer([32])),
    'bc4': tf.Variable(initializer([64])),
    'bc5': tf.Variable(initializer([32])),
    'out': tf.Variable(initializer([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
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
