import tensorflow as tf
import os 
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 100
display_step = 100

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X = tf.placeholder(tf.float32,[None,784])

En_h_1 = tf.Variable(tf.random_normal([28*28, 100]))
En_h_2 = tf.Variable(tf.random_normal([100, 10]))
De_h_1 = tf.Variable(tf.random_normal([10, 100]))
De_h_2 = tf.Variable(tf.random_normal([100, 28*28]))

En_b_1 = tf.Variable(tf.random_normal([100]))
En_b_2 = tf.Variable(tf.random_normal([10]))
De_b_1 = tf.Variable(tf.random_normal([100]))
De_b_2 = tf.Variable(tf.random_normal([28*28]))

def encoder(x):
    En_z_1 = tf.add(tf.matmul(x, En_h_1), En_b_1)
    En_z_1 = tf.nn.sigmoid(En_z_1)
    En_z_2 = tf.add(tf.matmul(En_z_1, En_h_2), En_b_2)
    En_z_2 = tf.nn.sigmoid(En_z_2)
    return En_z_2
    
def decoder(z):
    De_z_1 = tf.matmul(z, De_h_1) + De_b_1
    De_z_1 = tf.nn.sigmoid(De_z_1)
    De_z_2 = tf.matmul(De_z_1, De_h_2) + De_b_2
    De_z_2 = tf.nn.sigmoid(De_z_2)
    return De_z_2

Z = encoder(X)
loss = tf.losses.mean_squared_error(X, decoder(encoder(X)) )
# loss = tf.reduce_mean(tf.pow(X - decoder(encoder(X)), 2))
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
    
    for iter in range(30000):
        training_data, _ = mnist.train.next_batch(int(batch_size*iter/30000)*10+1)
        
        c_z, c_loss,_ = sess.run([Z, loss,opt], feed_dict={X:training_data})
        if iter % display_step == 0 or iter == 1:
            print('Step %i: Minibatch Loss: %f' % (iter, c_loss))
        
    pass
    
    print(c_z)
    
pass