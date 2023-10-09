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
    tr_dataset = tr_dataset.shuffle(train_size).batch(tr_bs, drop_remainder=True).repeat()
    ts_dataset = tf.data.Dataset.from_tensor_slices({'images':ts, 'labels':ts_y})
    ts_dataset = ts_dataset.shuffle(test_size).batch(ts_bs, drop_remainder=True).repeat()
    return(tr_dataset.as_numpy_iterator(), ts_dataset.as_numpy_iterator())
pass 

def CNN(amp = 1):
    x = tf.keras.Input([28, 28, 1])
    e1 = tf.keras.layers.Conv2D(16 * amp, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(x) #[14,14]
    e1 = tf.keras.layers.Conv2D(16 * amp, (3,3), padding="SAME", strides=(1,1), activation=tf.nn.relu)(e1) #[14,14]
    e1 = tf.keras.layers.Conv2D(16 * amp, (3,3), padding="SAME", strides=(1,1), activation=tf.nn.relu)(e1) #[14,14]
    e2 = tf.keras.layers.Conv2D(32 * amp, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(e1) #[7,7]
    e2 = tf.keras.layers.Conv2D(32 * amp, (3,3), padding="SAME", strides=(1,1), activation=tf.nn.relu)(e2) #[7,7]
    e2 = tf.keras.layers.Conv2D(32 * amp, (3,3), padding="SAME", strides=(1,1), activation=tf.nn.relu)(e2) #[7,7]
    e3 = tf.keras.layers.Conv2D(64 * amp, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(e2) #[4,4]
    e3 = tf.keras.layers.Conv2D(64 * amp, (3,3), padding="SAME", strides=(1,1), activation=tf.nn.relu)(e3) #[4,4]
    e3 = tf.keras.layers.Conv2D(64 * amp, (3,3), padding="SAME", strides=(1,1), activation=tf.nn.relu)(e3) #[4,4]
    
    fc = tf.keras.layers.Flatten()(e3)
    fc1 = tf.keras.layers.Dense(128 * amp, activation=tf.nn.relu)(fc)
    fc2 = tf.keras.layers.Dense(64 * amp, activation=tf.nn.relu)(fc1)
    fc3 = tf.keras.layers.Dense(32 * amp, activation=tf.nn.relu)(fc2)
    
    out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(fc3)
    
    return tf.keras.Model(inputs=x, outputs=out)
pass 

def ce(y, _y):
    return tf.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy()(y, _y))
pass 

def main():
    bs = 256
    tr_iter, ts_iter = get_mnist_iter(bs,bs)
    loss_stop_threshold = 5E-2
    amp = 2 # 神經網路參數放大倍率
    cnn = CNN(amp)
    
    def loss():
        fetcher = next(tr_iter)
        target, label = fetcher['images'], fetcher['labels']
        target = (target - 128.) / 2.
        label = tf.reshape(label, [-1])
        out = cnn(target)
        return ce(label, out)

    # training process 
    opt = tf.keras.optimizers.AdamW(learning_rate=1E-4, clipnorm=1)
    step = 0
    while loss() > loss_stop_threshold:
        step += 1
        opt.minimize(loss, var_list=cnn.trainable_weights)
        print('step:{} loss:{}'.format(step, loss().numpy()))



if __name__ == "__main__":
    main()
pass 