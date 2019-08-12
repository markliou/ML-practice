import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

learning_rate = 1E-5
batch_size = 32
iteration = 5000
beta = .25

def VQVAE(X, act=tf.nn.relu, dic_size=128):
    with tf.variable_scope('vqvae_e'):
        conv1e = tf.keras.layers.Conv2D(32, [3, 3], strides=2, padding='SAME', activation=act)(X)
        conv2e = tf.keras.layers.Conv2D(32, [3, 3], strides=2, padding='VALID', activation=act)(conv1e)
        conv3e = tf.keras.layers.Conv2D(32, [3, 3], strides=2, padding='SAME', activation=act)(conv2e)
        ze = tf.reshape(conv3e, [-1, conv3e.shape[1] * conv3e.shape[2], conv3e.shape[3]])
    
    with tf.variable_scope('vqvae_vq'):
        #crate the quantized vector dictionary 
        vq_dictionary = tf.Variable(tf.random.uniform([dic_size, conv3e.shape[1].value * conv3e.shape[2].value]), trainable=True, dtype=tf.float32, name='vq_dictionary')
        zq = tf.map_fn(lambda i:
                      tf.stack([ 
                                vq_dictionary[tf.argmin(tf.reduce_mean(tf.pow(j-vq_dictionary, 2), axis=-1))] for j in tf.unstack(i, axis=-1) 
                               ], axis=-1)
                      , ze, parallel_iterations=32)
    zq = ze + tf.stop_gradient(zq - ze)
    zq = tf.reshape(zq, [-1, conv3e.shape[1].value, conv3e.shape[2].value, conv3e.shape[3].value])
    with tf.variable_scope('vqvae_d'):
        conv1d = tf.keras.layers.Conv2DTranspose(32, [3, 3], strides=2, padding='VALID', activation=act)(zq)
        conv2d = tf.keras.layers.Conv2DTranspose(32, [3, 3], strides=2, padding='SAME', activation=act)(conv1d)
        conv3d = tf.keras.layers.Conv2DTranspose(32, [3, 3], strides=2, padding='SAME', activation=act)(conv2d)
    
    out = tf.keras.layers.Conv2D(1, [3, 3], strides=1, padding='SAME', activation=None)(conv3d)

    return [out, tf.reshape(ze, [-1, conv3e.shape[1].value, conv3e.shape[2].value, conv3e.shape[3].value]), zq]
pass


def main():
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    X_, VQVAE_ze, VQVAE_zq = VQVAE(X)

    # losses
    dec_loss = tf.reduce_mean(tf.pow(X - X_, 2)) #zq => X_
    vq_loss  = tf.reduce_mean(tf.pow((tf.stop_gradient(VQVAE_ze) - VQVAE_zq), 2)) * beta  #ze => zq
    enc_loss = tf.reduce_mean(tf.pow((VQVAE_ze - tf.stop_gradient(VQVAE_zq)), 2))   #X => zq
    
    # gradients for applying
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, centered=True, momentum=.9).minimize(dec_loss + vq_loss + enc_loss)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()

    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    for i in range(iteration):
        batch_x, _ = mnist.train.next_batch(batch_size)
        _, c_loss = sess.run([opt, dec_loss],feed_dict={X:np.reshape(batch_x, [-1, 28, 28, 1])})
        
        if i % 10 == 0 or i == 1:
            print('Step {}, Loss: {}'.format(i, c_loss))
        pass
    pass


    # Generate images via autoencoder
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        batch_x, _ = mnist.test.next_batch(2)
        batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
        g = sess.run(X_, feed_dict={X:batch_x})
        for j in range(2):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                                newshape=(28, 28, 3))
            a[j * 2][i].imshow(img)
            img = np.reshape(np.repeat(batch_x[j][:, :, np.newaxis], 3, axis=2),
                                newshape=(28, 28, 3))
            a[j * 2 + 1][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()

pass


if __name__=="__main__":
    main()
pass


