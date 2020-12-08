import tensorflow as tf 
import numpy as np 

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

# create a dummy model
ex_cnn = cnn() 
ex_cnn.save('cnn.h5')
ex_cnn.save('/tmp/cnnpb')

# loading mode 
loaded_mode = tf.keras.models.load_model('cnn.h5')
print(loaded_mode.summary())

# INPUT_NODES = [n.op.name for n in loaded_mode.inputs]
# OUTPUT_NODES = [n.op.name for n in loaded_mode.outputs]
# print('== input information ==')
# print(INPUT_NODES)
# print('== output information ==')
# print(OUTPUT_NODES)

# freezing the model
# python /opt/intel/openvino_2021.1.110/deployment_tools/model_optimizer/mo.py --saved_model_dir /tmp/cnnpb/ -b 1


