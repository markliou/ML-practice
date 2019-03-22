import tensorflow as tf 
import numpy as np 
import json 
from PIL import Image

def g_noise(pic, scale = .03):
    noise = tf.random.normal(tf.shape(pic), mean=0., stddev=255. * scale)
    return tf.cast(tf.clip_by_value((tf.cast(pic, tf.float32) + noise), 0, 255), tf.uint8)
pass

pic = tf.image.decode_jpeg(tf.read_file('1.jpg'))
pic = g_noise(pic, .1)
sess = tf.Session()
print(sess.run(tf.shape(pic)))
img = sess.run(pic)
img = Image.fromarray(img,"RGB")
img.save('tf_1.jpg')
sess.close()

jf = open('1.json')
j = json.load(jf)
# eye_details': {'look_vec'
jf.close()
print(eval(j['eye_details']['look_vec'])[:2])