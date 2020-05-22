import numpy as np
import tensorflow as tf

K_model = tf.keras.models.load_model('K_model.h5')
K_model.summary()