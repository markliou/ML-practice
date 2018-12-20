#####
# This script give an example for how to loading a PNG file directly using
# 1. feed image file name
# 2. Tensorflow loader 
# markliou 20181220
#####

import tensorflow as tf 
import numpy as np

img_name = tf.placeholder(dtype=tf.string)
img_str = tf.read_file(img_name)
img = tf.image.decode_png(img_str)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c_img = sess.run(img, feed_dict={img_name:'unnamed.png'})
    print(c_img)
    print(c_img.shape)
