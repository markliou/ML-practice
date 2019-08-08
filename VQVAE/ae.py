import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

learning_rate = 1E-4
batch_size = 32
iteration = 5000

def AE(X, act=tf.nn.relu):
    conv1e = tf.keras.layers.Conv2D(32, [3, 3], strides=2, padding='SAME', activation=act)(X)
    conv2e = tf.keras.layers.Conv2D(32, [3, 3], strides=2, padding='VALID', activation=act)(conv1e)
    conv3e = tf.keras.layers.Conv2D(32, [3, 3], strides=2, padding='SAME', activation=act)(conv2e)
    conv1d = tf.keras.layers.Conv2DTranspose(32, [3, 3], strides=2, padding='VALID', activation=act)(conv3e)
    conv2d = tf.keras.layers.Conv2DTranspose(32, [3, 3], strides=2, padding='SAME', activation=act)(conv1d)
    conv3d = tf.keras.layers.Conv2DTranspose(32, [3, 3], strides=2, padding='SAME', activation=act)(conv2d)
    
    out = tf.keras.layers.Conv2D(1, [3, 3], strides=1, padding='SAME', activation=None)(conv3d)

    return out
pass


def main():
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    X_ = AE(X)
    loss = tf.reduce_mean(tf.pow(X - X_, 2))
    
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

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
        _, c_loss = sess.run([opt, loss],feed_dict={X:np.reshape(batch_x, [-1, 28, 28, 1])})
    
        if i % 1000 == 0 or i == 1:
            print('Step {}, Loss: {}'.format(i, c_loss))
        pass
    pass


    # Generate images via autoencoder
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        batch_x, _ = mnist.test.next_batch(4)
        g = sess.run(X_, feed_dict={X:np.reshape(batch_x, [-1, 28, 28, 1])})
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                                newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()

pass


if __name__=="__main__":
    main()
pass


