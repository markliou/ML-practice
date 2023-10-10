import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np 

def get_mnist_iter(tr_bs=32, ts_bs=32):
    train_size = 60000
    test_size = 10000
    (tr, tr_y), (ts, ts_y) = tf.keras.datasets.fashion_mnist.load_data()
    tr, ts = tf.reshape(tr, [-1, 28, 28, 1]), tf.reshape(ts, [-1, 28, 28, 1])
    tr_y, ts_y = tf.reshape(tr_y, [-1, 1]), tf.reshape(ts_y, [-1, 1])
    tr_dataset = tf.data.Dataset.from_tensor_slices({'images':tr, 'labels':tr_y})
    tr_dataset = tr_dataset.shuffle(train_size).batch(tr_bs, drop_remainder=True).repeat()
    ts_dataset = tf.data.Dataset.from_tensor_slices({'images':ts, 'labels':ts_y})
    ts_dataset = ts_dataset.shuffle(test_size).batch(ts_bs, drop_remainder=True).repeat()
    return(tr_dataset.as_numpy_iterator(), ts_dataset.as_numpy_iterator())
pass 

def CNN(amp = 1):
    initializer = tf.keras.initializers.RandomNormal()
    x = tf.keras.Input([28, 28, 1])
    
    fc = tf.keras.layers.Flatten()(x)
    fc1 = tf.keras.layers.Dense(16 * amp ,kernel_initializer=initializer, bias_initializer=initializer, activation=tf.nn.relu)(fc)
    fc2 = tf.keras.layers.Dense(32 * amp ,kernel_initializer=initializer, bias_initializer=initializer, activation=tf.nn.relu)(fc1)
    
    out = tf.keras.layers.Dense(10,kernel_initializer=initializer, bias_initializer=initializer, activation=tf.nn.softmax)(fc2)
    
    return tf.keras.Model(inputs=x, outputs=out)
pass 

def ce(y, _y):
    return tf.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy()(y, _y))
pass 

def main():
    bs = 1024
    tr_iter, ts_iter = get_mnist_iter(bs, 10000)
    loss_stop_threshold = 1E-3
    amp = 1 # 神經網路參數放大倍率，建議使用1、10、100比較容易看出差距
    cnn = CNN(amp)
    early_stop = 10
    
    # try to train the model to overfitting
    fetcher = next(tr_iter)
    target, label = fetcher['images'], fetcher['labels']
    target = (target - 128.) / 256.
    label = tf.reshape(label, [-1])
    
    @tf.function
    def loss():
        out = cnn(target)
        return ce(label, out)

    # training process 
    opt = tf.keras.optimizers.AdamW(learning_rate=1E-4, clipnorm=1)
    step = 0
    # while early_stop > 0:
    while step < 5000:
        step += 1
        opt.minimize(loss, var_list=cnn.trainable_weights)
        c_loss = loss().numpy()
        if step % 100 == 0:
            print('step:{} loss:{}'.format(step, c_loss))
        if c_loss < loss_stop_threshold:
            early_stop -= 1
        
    # test process
    def show_acc(target, label):
        label = tf.reshape(label, [-1])
        out = tf.math.argmax(cnn(target), axis=-1)
        acc = tf.keras.metrics.Accuracy()
        acc.update_state(out, label)
        print("accuracy:{}".format(acc.result().numpy()))
    
    print("training:")
    show_acc(target, label)
    
    fetcher = next(ts_iter)
    target, label = fetcher['images'], fetcher['labels']
    target = (target - 128.) / 256.
    print("test:")
    show_acc(target, label)



if __name__ == "__main__":
    main()