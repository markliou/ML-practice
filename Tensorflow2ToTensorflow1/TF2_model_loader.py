import numpy as np
import tensorflow as tf

from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework import graph_io

def mlp():
    features = tf.keras.Input(shape=[90]) 
    h1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(features * .2)
    h2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(h1)
    h3 = tf.keras.layers.Dense(32, activation=tf.nn.relu)(h2)
    h4 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(h3)
    h5 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(h4)
    out = tf.keras.layers.Dense(1, activation=None)(h5)
    return tf.keras.Model(inputs=features, outputs=out, name='mlp')
pass
K_model = mlp()
print(K_model.summary())

K_model.load_weights('K_model.h5')
print('model loading success !!')

## choose the output node for .pb
# for n in tf.get_default_graph().as_graph_def().node:
#     print(n.name)
# exit()
## finally we get : dense_5/BiasAdd

## converting part
sess = tf.Session()
sess.run(tf.global_variables_initializer())
frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["dense_5/BiasAdd"])
graph_io.write_graph(frozen, './', 'K_model_tf1.pb', as_text=False)