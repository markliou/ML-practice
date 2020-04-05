import tensorflow as tf
import numpy as np

# Training parameters.
learning_rate = 0.001
training_steps = 100
batch_size = 128
display_step = 10
model_path = './'

# define the dataset
(tr_x, tr_y), (ts_x, ts_y) = tf.keras.datasets.mnist.load_data()
(tr_x, tr_y), (ts_x, ts_y) = (tr_x/255., tr_y), (ts_x/255., ts_y)

tr = tf.data.Dataset.from_tensor_slices((tr_x.astype(np.float32), tr_y))
tr = tr.repeat().shuffle(100).batch(batch_size).prefetch(3)

# construct the model with K api
def cnn():
    cnn_input = tf.keras.Input(shape=[28, 28])
    cnn0 = tf.expand_dims(cnn_input, axis=-1, name='input_channel')
    cnn1 = tf.keras.layers.Conv2D(64, 3, strides=(1,1), activation=tf.nn.relu)(cnn0)
    cnn2 = tf.keras.layers.Conv2D(64, 3, strides=(2,2), activation=tf.nn.relu)(cnn1)
    cnn3 = tf.keras.layers.Conv2D(128, 3, strides=(1,1), activation=tf.nn.relu)(cnn2)
    cnn4 = tf.keras.layers.Conv2D(128, 3, strides=(2,2), activation=tf.nn.relu)(cnn3)
    fc0 = tf.keras.layers.Flatten()(cnn4)
    fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(fc0)
    fc2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(fc1)
    out = tf.keras.layers.Dense(10)(fc2)

    return tf.keras.Model(inputs=cnn_input, outputs=out, name='cnn')
pass

cnn_model = cnn()
print(cnn_model.summary())
print(cnn_model)

# training loop
tr_iter = iter(tr)
opt = tf.keras.optimizers.RMSprop(learning_rate, centered=False)
for step in range(training_steps):
    def loss():
        tr_x_c, tr_y_c = tr_iter.__next__()
        cnn_logits = cnn_model(tr_x_c)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(tr_y_c, dtype=tf.int64), logits=cnn_logits))
    pass
    opt.minimize(loss, cnn_model.trainable_weights)
    if step % display_step == 0:
        print(loss())
    pass
cnn_model.save(model_path) # the pb will be saved as 'saved_model.pb'

# evaluate the accuracy
ts_y_pred = cnn_model(ts_x.astype(np.float32))
correct_predictions = tf.equal(tf.argmax(ts_y_pred, 1), tf.cast(ts_y, tf.int64))
acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=-1)
print('test accuracy:{}'.format(acc))

## convert to tensorflow lite
tfl_converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
cnn_model_tfl = tfl_converter.convert()
open("cnn.tflite", "wb").write(cnn_model_tfl)

## note ##
# this would be similar to use the CMD api such as :
# tflite_convert --saved_model_dir=`pwd` --output_file=cnn.tflite

# the quantizing model is also tried here
tfl_converter.optimizations = [tf.lite.Optimize.DEFAULT]
qcnn_model_tfl = tfl_converter.convert()
open("qcnn.tflite", "wb").write(qcnn_model_tfl)