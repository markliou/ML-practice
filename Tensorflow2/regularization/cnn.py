import tensorflow as tf
import tensorflow.python.framework.ops as tfop
import numpy as np
# print(tf.__version__)

# if the eager is not nessessary, turn off it.
# tfop.disable_eager_execution()

def cnn_model():
    REG = tf.keras.regularizers.L2(l2=1e-4)

    x = tf.keras.layers.Input(shape=(32, 32, 3)) # the input operator of keras is different from Placeholder of TF1
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=tf.nn.relu, kernel_regularizer=REG)(x)
    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation=tf.nn.relu, kernel_regularizer=REG)(conv1)
    conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation=tf.nn.relu, kernel_regularizer=REG)(conv2)
    conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation=tf.nn.relu, kernel_regularizer=REG)(conv3)
    # print(conv4)
    flatten = tf.keras.layers.Flatten()(conv4)
    # print(flatten)
    fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_regularizer=REG)(flatten)
    fc2 = tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=REG)(fc1) 
    out = tf.keras.layers.Dense(10, activation=None, kernel_regularizer=REG)(fc2) 
    # print(out)

    return tf.keras.Model(inputs=x, outputs=out) 
pass


##### directly use the keras model object #####
# call the function which give an model object
c_cnn_model = cnn_model()


### manipulating dataset
(cifar10_img_tr, cifar10_y_tr), (cifar10_img_ts, cifar10_y_ts) = tf.keras.datasets.cifar10.load_data()

# creating tf dataset, the cifar10 is the raw data which needed to be preprocessed
tr_dataset = tf.data.Dataset.from_tensor_slices({'img':cifar10_img_tr, 'lab':cifar10_y_tr})
tr_dataset = tr_dataset.map(lambda x: {'img':(tf.cast(x['img'], dtype=tf.float32) - 128)/128, 'lab':x['lab']})
tr_dataset = tr_dataset.batch(32)
tr_dataset = tr_dataset.prefetch(4)
tr_dataset = tr_dataset.cache()
tr_dataset = tr_dataset.repeat()
tr_dataset = tr_dataset.shuffle(buffer_size=10000)

### setting the training parameters
learning_rate = 1E-5
iteration_num = 5000

Optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

### optimization the mode
print('optimizing stage ...')
for iteration_step in range(iteration_num):
    tr_dataset_fetcher = next(iter(tr_dataset))
    img_fetcher, lab_fetcher = tr_dataset_fetcher['img'], tf.one_hot(tf.reshape(tr_dataset_fetcher['lab'], shape=[-1,]), depth=10, on_value=1, off_value=0)
    
    def loss_m():
    # we still need to put the model into this callable function for catching the variables the model using
        predicts = c_cnn_model(img_fetcher)
        # since the tensorflow's softmax_cross_entropy_with_logits caused some problems (such as the strange abnormal values),
        # we also implement this function by hand.
        batch_loss = tf.reduce_mean(-tf.reduce_sum(tf.cast(lab_fetcher, dtype=tf.float32) * tf.math.log(tf.nn.softmax(predicts)+1e-15), axis=-1), axis=-1)
        regularization = tf.reduce_mean(c_cnn_model.losses)
        print('batch loss:{}'.format(batch_loss))
        print('regularization:{}'.format(regularization))
        return batch_loss + regularization
               
    Optimizer.minimize(loss_m, c_cnn_model.trainable_variables) # the loss for minimize method should be a function to calling

    print('step:{} loss:{}'.format(iteration_step, loss_m()))
    # exit()




