import tensorflow as tf
import numpy as np
# print(tf.__version__)

def cnn_model():
    x = tf.keras.layers.Input([32, 32, 3]) # the input operator of keras is different from Placeholder of TF1
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=tf.nn.relu)(x)
    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation=tf.nn.relu)(conv1)
    conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation=tf.nn.relu)(conv2)
    conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation=tf.nn.relu)(conv3)
    # print(conv4)
    flatten = tf.keras.layers.Flatten()(conv4)
    # print(flatten)
    fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(flatten)
    fc2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc1) 
    out = tf.keras.layers.Dense(10, activation=None)(fc2) 
    # print(out)

    return tf.keras.Model(inputs=x, outputs=out) 
pass


##### directly use the keras model object #####
# call the function which give an model object
c_cnn_model = cnn_model()

# show the information about the model
print(c_cnn_model.summary())
tf.keras.utils.plot_model(c_cnn_model, 'cnn.png', show_shapes=True)

# inferencing
print(c_cnn_model(np.random.random([10, 32, 32, 3])))

# saving the model
tf.saved_model.save(c_cnn_model, "./c_cnn_model/") # saving the model. The .pb is made automatically.

# try the model loader
loader = tf.saved_model.load("./c_cnn_model/") 
print('loadered...')
## the dedault output of the TF2 will set as a concrete function with signature 'serving_default'
print(loader.signatures["serving_default"](tf.ones([10, 32, 32, 3])))

### manipulating dataset
(cifar10_img_tr, cifar10_img_ts), (cifar10_y_tr, cifar10_y_ts) = tf.keras.datasets.cifar10.load_data()
print(np.array(cifar10_img_tr).shape)


