import tensorflow as tf 
import numpy as np 

batch_size = 32
iter_no = 50000
learning_rate = 1e-3
display_step = 10

features, labels = [], []
# k_file = open('dataset_20200409.tab')
k_file = open('tr.tab')
for line in k_file.readlines():
    line = line.rstrip()
    contents = line.split("\t")
    label = contents.pop()
    labels.append([float(label)])
    features.append([float(i) for i in contents])
pass 

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

K_dataset = tf.data.Dataset.from_tensor_slices({"features":np.array(features).astype(np.float32), "labels":np.array(labels).astype(np.float32)})
K_dataset = K_dataset.repeat().shuffle(100).batch(32).prefetch(3)
K_dataset_iter = iter(K_dataset)
opt = tf.keras.optimizers.RMSprop(learning_rate, centered=True)
for step in range(iter_no):
    c_data = K_dataset_iter.__next__()
    def loss():
        pred = K_model(c_data["features"])
        return tf.reduce_mean(tf.pow((pred - c_data["labels"]), 2))
    pass 
    opt.minimize(loss, K_model.trainable_weights)
    if step % display_step == 0:
        print("step:{} loss:{}".format(step, loss().numpy()))
    pass

# K_model.save('K_model/')
K_model.save('K_model.h5')
# tf.saved_model.save(K_model, 'K_model')

