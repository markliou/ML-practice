import tensorflow as tf
import os 
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 100
display_step = 100

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# tf.conv2d => [filter_height, filter_width, in_channels, out_channels]
En_h_1 = tf.Variable(tf.random_normal([3, 3, 1, 100]))
En_h_2 = tf.Variable(tf.random_normal([3, 3, 100, 10]))
De_h_1 = tf.Variable(tf.random_normal([3, 3, 100, 10]))
De_h_2 = tf.Variable(tf.random_normal([3, 3, 1, 100]))

En_b_1 = tf.Variable(tf.random_normal([100]))
En_b_2 = tf.Variable(tf.random_normal([10]))
De_b_1 = tf.Variable(tf.random_normal([100]))
De_b_2 = tf.Variable(tf.random_normal([1]))

def encoder(x):
    # tf.nn.conv2d(
        # input,
        # filter,
        # strides,
        # padding,
        # use_cudnn_on_gpu=True,
        # data_format='NHWC',
        # dilations=[1, 1, 1, 1],
        # name=None
    # )
    En_f_1 = tf.nn.conv2d(x, En_h_1, strides=[1, 2, 2, 1], padding='SAME') + En_b_1
    En_f_1 = tf.nn.sigmoid(En_f_1)
    En_f_2 = tf.nn.conv2d(En_f_1, En_h_2, strides=[1, 2, 2, 1], padding='SAME') + En_b_2
    En_f_2 = tf.nn.sigmoid(En_f_2)
    
    return En_f_2
    
def decoder(z):
    # tf.nn.conv2d_transpose(
        # value, => shape [batch, height, width, in_channels] for NHWC
        # filter, => [height, width, output_channels, in_channels]
        # output_shape,
        # strides,
        # padding='SAME',
        # data_format='NHWC',
        # name=None
    # )
    c_batch_size = tf.shape(z)[0]  
    De_f_1 = tf.nn.conv2d_transpose(z, De_h_1, output_shape=[c_batch_size, 14, 14, 100], strides=[1, 2, 2,1]) + De_b_1
    De_f_1 = tf.nn.sigmoid(De_f_1)
    De_f_2 = tf.nn.conv2d_transpose(De_f_1, De_h_2, output_shape=[c_batch_size, 28, 28, 1], strides=[1, 2, 2, 1]) + De_b_2
    De_f_2 = tf.nn.sigmoid(De_f_2)
    
    return De_f_2

X = tf.placeholder(tf.float32,[None, 28, 28, 1]) # NHWC
Z = encoder(X)
Y = decoder(Z)
loss = tf.losses.mean_squared_error( X, Y)
opt = tf.train.AdamOptimizer(0.001).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    # training_data, _ = mnist.train.next_batch(batch_size)
    # print(training_data.shape)
    # er = sess.run(er, feed_dict={X:training_data})
    # print(er.shape)
    # exit()
    
    for iter in range(10000):
        # c_batch_size = int(batch_size*iter/30000)*10+1
        c_batch_size = batch_size
        training_data, _ = mnist.train.next_batch(c_batch_size)
        training_data = np.reshape(training_data, [c_batch_size, 28, 28, 1])
        
        c_z, c_loss, _ = sess.run([Z, loss, opt], feed_dict={X:training_data})
        if iter % display_step == 0 or iter == 1:
            print('Step %i: Minibatch Loss: %f' % (iter, c_loss))
        
    pass
    
    print(c_z)

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        batch_x = np.reshape(batch_x, [n, 28, 28, 1])
        # Encode and decode the digit image
        g = sess.run( Y, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
    
    
pass
