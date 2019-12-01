import tensorflow as tf 
import numpy as np 
# import cv2 

# The basic idea of the TFrecorder is to use the binary format to enhancing the performace.
# 1. tf.data.Example: like the template. This will give the TFrecorder the format.
#    The auguments need the posional information. (e.g. specified the 'features')
# 2. tf.train.Features: Example module will store the data using the format encoded from Feature module
#    * the input shold be a python dictionary-like (key-value pair).
#    * the value shoule be the tf.train.<type> object as describe below
#    * all the input needed a position, e.g. specified the 'feature'
# 3. tf.train.Feature: convert the tf.train.<type> to the format that tf.train.Features accepting.
#    The feature is position specific (e.g. need specifying 'byte_list'). The type include
#    * bytes_list
#    * float_list
#    * int64_list
# 4. tf.train.BytesList, tf.train.Int64List, tf.train.FloatList : The 3 basic tensorflow dataformat. 
#    Feature module uses this kind of format. Some criteria should be noticed:
#    * the input should be a flat list
#    * all the input needed a position, e.g. specified the 'value'
# 5. The Example object should be written with the "string" into the file using tf.io.TFRecordWriter. 
#    Before writing, the Example need to transform to string with Example.SerializeToString method.
#
# reference:
# https://zhuanlan.zhihu.com/p/40588218
# https://www.jianshu.com/p/78467f297ab5
# https://blog.gtwang.org/programming/tensorflow-read-write-tfrecords-data-format-tutorial/
# https://www.tensorflow.org/tutorials/load_data/tfrecord

# set the basic parameters
pics = ['1.jpg', '2.jpg']
# pics = ['2.jpg']
lables = [0, 1]

tfrecorder_name = 'pic.tfr'

sess = tf.InteractiveSession()
writer = tf.io.TFRecordWriter(tfrecorder_name)
for i in range(len(pics)):
    ##### read image
    tf_pic = tf.io.decode_jpeg(tf.io.read_file(pics[i]))
    pic_con = np.array(sess.run(tf_pic))
    # print(sess.run(tf_pic))
    # print(tf_pic)
    # print(pic_con.shape)

    ##### create an Example-accepted data structure
    feature={
              'pic'     : tf.train.Feature(int64_list=tf.train.Int64List(value=np.reshape(pic_con, [-1]).tolist()))
             ,'size'    : tf.train.Feature(int64_list=tf.train.Int64List(value=list(pic_con.shape[0:2])))
             ,'channel' : tf.train.Feature(int64_list=tf.train.Int64List(value=[pic_con.shape[2]]))
             ,'shape'   : tf.train.Feature(int64_list=tf.train.Int64List(value=list(pic_con.shape[:])))
             ,'label'   : tf.train.Feature(int64_list=tf.train.Int64List(value=[lables[i]]))
            }
    # print(feature)
    
    ##### make the Example object
    sample = tf.train.Example(
                              features=tf.train.Features(feature=feature)
                             )
    # print(tf.train.Features(feature={'pic':tf.train.Int64List(value=np.reshape(pic_con, [-1]).tolist()) }))
    # print(tf.train.Int64List(value=np.reshape(pic_con, [-1]).tolist()))
    # print(feature)
    # print(sample)

    ##### write the Example objects into a file
    # print(sample)
    # print(sample.SerializeToString())
    # print(tf.train.Example.FromString(sample.SerializeToString())) # check it this can be revered by FromString method
    writer.write(sample.SerializeToString())

pass
sess.close()