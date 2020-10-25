import numpy as np 
import tensorflow as tf # use the tf2.4
import tensorflow_datasets as tfds 

inner_lr = 1E-3
outter_lr = 1E-3
inner_task_loop_no = 32 # give 32 tasks of 2ways-1shot
opt_inner = tf.keras.optimizers.RMSprop(inner_lr)
opt_outter = tf.keras.optimizers.RMSprop(outter_lr)

def cnn():
    Input = tf.keras.Input([28, 28, 1])
    Input_n = Input/128.0 - 1
    conv1 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(Input_n) #[14,14]
    conv2 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(conv1) #[7,7]
    conv3 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(conv2) #[4,4]
    fc = tf.keras.layers.Flatten()(conv3)
    fc1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc)
    fc2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc1)
    fc3 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc2)
    out = tf.keras.layers.Dense(1, activation=None)(fc3)
    return tf.keras.Model(inputs=Input, outputs=out)
pass 

(kmnist_tr, kmnist_ts) = tfds.load('kmnist', 
                                   split=['train','test'],
                                   shuffle_files=True,
                                   as_supervised=True,
                                   with_info=False,)

# Task hint:
# this example will use 2ways-1shot for training the MAML.
# the inner task loop will set as 30. 
kmnist_tr = kmnist_tr.batch(2).repeat()
kmnist_tr_iter = iter(kmnist_tr)
meta_labs = tf.Variable([[1], [0]], dtype=tf.float32)

cnn_model = cnn()

# according to the TF2 manual, the variable of the model would need to be
# initialized, and use it to inferece something is a way to initialize 
# the varialbe.
imgs, labs = next(kmnist_tr_iter)
_ = cnn_model(imgs)
loss_inner = lambda: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=meta_labs, logits=cnn_model(imgs)))

for step in range(5000):
    # keep the original weights for meta-training
    meta_weights = [tf.Variable(target_weights) for target_weights in cnn_model.trainable_weights]
    
    def meta_loss():
        loss_outter = 0
        for task_count in range(inner_task_loop_no):
            # fetch the dataset. The first dataset is always be 1. 
            # So, the labels will be defined as the meta-labels insteadly. 
            imgs, labs = next(kmnist_tr_iter)
            
            # put the meta-weights into the model
            for model_weight_index in range(len(cnn_model.trainable_weights)):
                cnn_model.trainable_weights[model_weight_index].assign(meta_weights[model_weight_index])
            pass

            # updateing the weights from meta-weights
            opt_inner.minimize(loss=loss_inner, var_list=cnn_model.trainable_weights) 
            
            # calculating the task loss (after the meta-weights updating)
            loss_outter += loss_inner()
        pass 
        return loss_outter/inner_task_loop_no
    pass 

    # put the meta-weights into the model
    for model_weight_index in range(len(cnn_model.trainable_weights)):
        cnn_model.trainable_weights[model_weight_index].assign(meta_weights[model_weight_index])
    pass

    # update the meta-weights
    opt_outter.minimize(loss=meta_loss, var_list=cnn_model.trainable_weights) 

    print("step: {}  outter_loss: {}".format(step, meta_loss()))
pass 
