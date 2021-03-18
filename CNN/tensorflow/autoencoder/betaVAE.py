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
    mean_out = tf.keras.layers.Conv2D(128, (3,3), padding="SAME", strides=(2,2), activation=None)(e3) #[2,2]
    logvar_out = tf.keras.layers.Conv2D(128, (3,3), padding="SAME", strides=(2,2), activation=None)(e3) #[2,2]
    return tf.keras.Model(inputs=x, outputs=[mean_out, logvar_out])
pass 

def AE_decoder():
    x = tf.keras.Input([2, 2, 128])
    d1 = tf.keras.layers.Conv2DTranspose(128, (5,5), padding="VALID", strides=(2,2), activation=tf.nn.relu)(x) #[7,7]
    d2 = tf.keras.layers.Conv2DTranspose(64, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(d1) #[14,14]
    out = tf.keras.layers.Conv2DTranspose(1, (3,3), padding="SAME", strides=(2,2), activation=None)(d2) #[28,28]
    return tf.keras.Model(inputs=x, outputs=out)
pass 

def latent_resampling(mean, logvar):
    # 一開始從 encoder 取得的數並不一定符合標準常態分布。
    # 但如果近似常態分佈的分布的抽樣都可以透過修改從常態分佈中取的的亂數，
    # 重新按照mean跟var來重新抽取一個對應的數字。
    # 因此只需要在這邊重新抽樣，並看原本抽樣的結果跟normal distibution差異有多少來設計loss就行了。
    # 重新抽樣的部分可以參考: https://www.probabilitycourse.com/chapter4/4_2_3_normal.php

    # 先從常態分布抽樣
    eps = tf.random.normal(shape=mean.shape)
    # 重新依照由encoder的var來進行抽樣。因為重新抽樣是從常態分佈抽出來的，所以要依照encoder的分布重新計算。
    # 不過因為希望ecoder是給logvar(loss比較好算)，所以先取exp算回來。
    r_logvar = eps * tf.exp(logvar * .5)
    return mean + r_logvar

def mse(y, _y):
    return tf.reduce_mean((y - _y)**2)
pass 

def log_nomral_pdf(z, mean, logvar):
    # f_z = tf.keras.layers.Flatten()(z)
    # f_mean = tf.keras.layers.Flatten()(mean)
    # f_logvar = tf.keras.layers.Flatten()(logvar)
    log2pi = tf.math.log(2. * np.pi)
    log_pdf = -.5 * (tf.pow((z - mean), 2.) * tf.exp(-logvar) + logvar + log2pi)
    return tf.reduce_sum(log_pdf)
pass

def main():
    tr_iter, ts_iter = get_mnist_iter()
    beta = 5
    en = AE_encoder()
    de = AE_decoder()
    
    def loss():
        target = next(tr_iter)['images']
        target = (target - 128.) / 2.
        latent_mean, latent_logvar = en(target)
        latent = latent_resampling(latent_mean, latent_logvar)
        out = de(latent)
        e_pdf = log_nomral_pdf(latent, latent_mean, latent_logvar)
        s_pdf = log_nomral_pdf(latent, 0., 0.)
        log_normal_pdf_loss = tf.reduce_mean((e_pdf - s_pdf))
        return mse(target, out) + beta * log_normal_pdf_loss
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