import tensorflow as tf
# import tensorflow.python.framework.ops as tfop
import numpy as np
# print(tf.__version__)

# if the eager is not nessessary, turn off it.
# tfop.disable_eager_execution()

def cnn_model():
    x = tf.keras.layers.Input(shape=(32, 32, 3)) # the input operator of keras is different from Placeholder of TF1
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=tf.nn.elu)(x)
    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation=tf.nn.elu)(conv1)
    conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation=tf.nn.elu)(conv2)
    conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation=tf.nn.elu)(conv3)
    # print(conv4)
    flatten = tf.keras.layers.Flatten()(conv4)
    # print(flatten)
    fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(flatten)
    for i in range(100):
        fc1 = tf.keras.layers.Dense(256, activation=tf.nn.elu)(fc1)
    fc2 = tf.keras.layers.Dense(128, activation=tf.nn.elu)(fc1) 
    out = tf.keras.layers.Dense(10, activation=None, kernel_initializer=tf.random_normal_initializer(mean=.0, stddev=.99))(fc2) 
    # print(out)

    return tf.keras.Model(inputs=x, outputs=out) 
pass


##### directly use the keras model object #####
# call the function which give an model object
c_cnn_model = cnn_model()
w_cnn_model = cnn_model()

# show the information about the model
print(c_cnn_model.summary())
tf.keras.utils.plot_model(c_cnn_model, 'cnn.png', show_shapes=True)

# inferencing
print(c_cnn_model(np.random.random([10, 32, 32, 3])))

# ###### saving the model
# tf.saved_model.save(c_cnn_model, "./c_cnn_model/") # saving the model. The .pb is made automatically.

# ###### try the model loader
# loader = tf.saved_model.load("./c_cnn_model/") 
# print('loadered...')
# ## the dedault output of the TF2 will set as a concrete function with signature 'serving_default'
# print(loader.signatures["serving_default"](tf.ones([10, 32, 32, 3])))

### manipulating dataset
(cifar10_img_tr, cifar10_y_tr), (cifar10_img_ts, cifar10_y_ts) = tf.keras.datasets.cifar10.load_data()
# print(np.array(cifar10_img_tr).shape)
# print(cifar10_img_tr[0])
# exit()

# creating tf dataset, the cifar10 is the raw data which needed to be preprocessed
tr_dataset = tf.data.Dataset.from_tensor_slices({'img':cifar10_img_tr, 'lab':cifar10_y_tr})
tr_dataset = tr_dataset.map(lambda x: {'img':(tf.cast(x['img'], dtype=tf.float32) - 128)/128, 'lab':x['lab']})
tr_dataset = tr_dataset.batch(32)
tr_dataset = tr_dataset.prefetch(4)
tr_dataset = tr_dataset.cache()
tr_dataset = tr_dataset.repeat()
tr_dataset = tr_dataset.shuffle(buffer_size=10000)
# print(tr_dataset)
# exit()

## if you use the eager execution, turn the dataset to iterator would be conveneient for the next steps
# tr_dataset_iter = tr_dataset.as_numpy_iterator()
# print(tr_dataset_iter.__next__())
# exit()

### setting the training parameters
learning_rate = 1E-5
warming_iter = 1000
iteration_num = 500000

Optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, centered=True)

for warming_iter in range(500):
    tr_dataset_fetcher = next(iter(tr_dataset))
    img_fetcher, lab_fetcher = tr_dataset_fetcher['img'], tf.one_hot(tf.reshape(tr_dataset_fetcher['lab'], shape=[-1,]), depth=10, on_value=1, off_value=0)
    def loss_w():
    # since the tensorflow's softmax_cross_entropy_with_logits caused some problems (such as the strange abnormal values),
    # we also implement this function by hand.
        in1 = np.reshape(np.random.normal(0, 1, 32 * 32 * 32 * 3), [32, 32, 32, 3])
        in2 = np.reshape(np.random.normal(0, 1, 32 * 32 * 32 * 3), [32, 32, 32, 3])
        in3 = (in1  + in2  )*.5
        predict1 = w_cnn_model(in1)
        predict2 = w_cnn_model(in2)
        predict3 = w_cnn_model(in3)
        predict2_oh = tf.cast(tf.one_hot(tf.argmax(predict2, axis=-1), depth=10, on_value=1, off_value=0), dtype=tf.float32)
        predict3_oh = tf.cast(tf.one_hot(tf.argmax(predict3, axis=-1), depth=10, on_value=1, off_value=0), dtype=tf.float32)
        #print(predict2_oh)
        #return tf.reduce_mean(-tf.reduce_sum(tf.cast(lab_fetcher, dtype=tf.float32) * tf.math.log(tf.nn.softmax(predicts)+1e-15), axis=-1), axis=-1)
        #return tf.reduce_mean(tf.reduce_sum(-tf.nn.softmax(predict2) * tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1))
        #return tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(predict3) * tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1))
        return tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(predict1) * tf.math.log(tf.nn.softmax(predict3)+1e-15), axis=-1)) + tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(predict2) * tf.math.log(tf.nn.softmax(predict3)+1e-15), axis=-1)) + tf.reduce_mean(tf.reduce_sum(-tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1))
        #return tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(predict3) * tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1)) - .1 * tf.reduce_mean(tf.reduce_sum(-tf.nn.softmax(predict2) * tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1))
        #return tf.reduce_mean(-tf.reduce_sum(predict3_oh * tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1)) - 1 * tf.reduce_mean(-tf.reduce_sum(predict2_oh * tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1))
        #return tf.reduce_mean(tf.reduce_sum(-tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1))
        #return tf.reduce_mean(tf.reduce_mean(-tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1)) + tf.reduce_mean(-tf.reduce_sum(predict3_oh * tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1)) - tf.reduce_mean(-tf.reduce_sum(predict2_oh * tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1)) * .1
        #return -tf.abs(tf.reduce_mean(tf.reduce_mean(-tf.math.log(tf.nn.softmax(predict1)+1e-15), axis=-1)) - tf.reduce_mean(tf.reduce_mean(-tf.math.log(tf.nn.softmax(predict2)+1e-15), axis=1)))
    Optimizer.minimize(loss_w, w_cnn_model.trainable_variables)
    print('warming step:{} loss:{}'.format(warming_iter, loss_w()))



### optimization the mode
print('optimizing stage ...')
for iteration_step in range(iteration_num):
    tr_dataset_fetcher = next(iter(tr_dataset))
    img_fetcher, lab_fetcher = tr_dataset_fetcher['img'], tf.one_hot(tf.reshape(tr_dataset_fetcher['lab'], shape=[-1,]), depth=10, on_value=1, off_value=0)

    ## method 2: using the minimize of the optimizer
    def loss_m():
        predicts = c_cnn_model(img_fetcher)
        return tf.reduce_mean(-tf.reduce_sum(tf.cast(lab_fetcher, dtype=tf.float32) * tf.math.log(tf.nn.softmax(predicts)+1e-15), axis=-1), axis=-1)
    def loss_mw():
        predicts = w_cnn_model(img_fetcher)
        return tf.reduce_mean(-tf.reduce_sum(tf.cast(lab_fetcher, dtype=tf.float32) * tf.math.log(tf.nn.softmax(predicts)+1e-15), axis=-1), axis=-1)
    
    Optimizer.minimize(loss_m, c_cnn_model.trainable_variables) # the loss for minimize method should be a function to calling
    Optimizer.minimize(loss_mw, w_cnn_model.trainable_variables)

    # tensorboard
    c_loss_m = loss_m()
    c_loss_mw = loss_mw()
    tf.summary.scalar('c_loss_m', opt_loss)
    tf.summary.scalar('c_loss_mw', opt_loss)
    tfboa_merged = tf.summary.merge_all()
    tfboa_writer = tf.summary.FileWriter('Tensorboard/', sess.graph)


    print('step:{} c_loss_m:{} c_loss_mw:{}'.format(iteration_step, c_loss_m, c_loss_mw))
    # exit()




