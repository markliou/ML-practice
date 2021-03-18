import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np 

def get_mnist_iter(tr_bs=32, ts_bs=32):
    train_size = 60000
    test_size = 10000
    (tr, tr_y), (ts, ts_y) = tf.keras.datasets.mnist.load_data()
    tr, ts = tf.reshape(tr, [-1, 28, 28, 1]), tf.reshape(ts, [-1, 28, 28, 1])
    tr_y, ts_y = tf.reshape(tr_y, [-1, 1]), tf.reshape(ts_y, [-1, 1])
    tr_dataset = tf.data.Dataset.from_tensor_slices({'images':tr, 'labels':tr_y})
    tr_dataset = tr_dataset.shuffle(train_size).batch(tr_bs, drop_remainder=True)
    ts_dataset = tf.data.Dataset.from_tensor_slices({'images':ts, 'labels':ts_y})
    ts_dataset = ts_dataset.shuffle(test_size).batch(ts_bs, drop_remainder=True)
    return(tr_dataset.as_numpy_iterator(), ts_dataset.as_numpy_iterator())
pass 

def AE_encoder():
    x = tf.keras.Input([28, 28, 1])
    e1 = tf.keras.layers.Conv2D(16, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(x) #[14,14]
    e2 = tf.keras.layers.Conv2D(32, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(e1) #[7,7]
    e3 = tf.keras.layers.Conv2D(64, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(e2) #[4,4]
    out = tf.keras.layers.Conv2D(128, (3,3), padding="SAME", strides=(2,2), activation=None)(e3) #[2,2]
    return tf.keras.Model(inputs=x, outputs=out)
pass 

def AE_decoder():
    x = tf.keras.Input([2, 2, 128])
    d1 = tf.keras.layers.Conv2DTranspose(128, (5,5), padding="VALID", strides=(2,2), activation=tf.nn.relu)(x) #[7,7]
    d2 = tf.keras.layers.Conv2DTranspose(64, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(d1) #[14,14]
    out = tf.keras.layers.Conv2DTranspose(1, (3,3), padding="SAME", strides=(2,2), activation=None)(d2) #[28,28]
    return tf.keras.Model(inputs=x, outputs=out)
pass 


def main():
    tr_iter, ts_iter = get_mnist_iter()
    en = AE_encoder()
    de = AE_decoder()
    
    def loss():
        target = next(tr_iter)['images']
        target = (target - 128.) / 2.
        latent = en(target)
        out = de(latent)
        return tf.reduce_mean((target - out)**2)
    pass

    # training process 
    opt = tf.keras.optimizers.RMSprop(learning_rate=1E-4, clipnorm=1)
    for step in range(500):
        opt.minimize(loss, var_list=[en.trainable_weights, de.trainable_weights])
        print('step:{} loss:{}'.format(step, loss().numpy()))
    pass
pass 


if __name__ == "__main__":
    main()
pass 