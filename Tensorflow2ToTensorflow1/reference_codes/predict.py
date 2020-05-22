import numpy as np 
import tensorflow as tf

K_model = tf.keras.models.load_model('K_model.h5')
K_model.summary()

features, labels = [], []
# k_file = open('dataset_20200409.tab')
k_file = open('ts.tab')
for line in k_file.readlines():
    line = line.rstrip()
    contents = line.split("\t")
    label = contents.pop()
    labels.append([float(label)])
    features.append([float(i) for i in contents])
pass 

MAE = 0
for ins in range(len(labels)):
    pred = K_model(np.array([features[ins]]).astype(np.float32))
    MAE += abs(pred - labels[ins]) / len(labels)
pass 
print(MAE)

