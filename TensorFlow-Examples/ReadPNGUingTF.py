#####
# This script give an example for how to loading a PNG file directly using
# 1. feed image file name
# 2. Tensorflow loader 
# markliou 20181220
#####

import tensorflow as tf 
import numpy as np

# the file name can be directly passed via placeholder
img_name = tf.placeholder(dtype=tf.string)
img = tf.image.decode_png(tf.read_file(img_name + '.png'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c_img = sess.run(img, feed_dict={img_name:'unnamed'})
    print(c_img)
    print(c_img.shape)

# or try to pass a batch of file names
img_names = tf.placeholder(shape=[None], dtype=tf.string)
##remember to declare the dtype, because the original dtype is tf.string. The tf.map_fn will set the 
##tensor as tf.string defalutly. So decalre the dytpe is essential in this part.
imgs = tf.map_fn( lambda c_img_name: tf.image.decode_png(tf.read_file(c_img_name + '.png')), img_names, dtype=tf.uint8 )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c_imgs = sess.run(imgs, feed_dict={img_names:['unnamed', 'unnamed', 'unnamed']})
    print(c_imgs)
    print(c_imgs.shape)
    

