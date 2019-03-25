import tensorflow as tf 
import numpy as np 
import json 
import glob 


def get_look_vec_from_json(filename):
    with open(filename) as jf:
        j = json.load(jf)
    return eval(j['eye_details']['look_vec'])[:2]
pass 

def g_noise(pic, scale = .3):
    noise = tf.random.normal(tf.shape(pic), mean=0., stddev=255. * scale)
    return tf.cast(tf.clip_by_value((tf.cast(pic, tf.float32) + noise), 0, 255), tf.uint8)
pass

def get_gaze_pic(filename):
    # JPG format. Output : [HWC]
    # return (tf.image.decode_jpeg(tf.read_file(filename)))
    return g_noise(tf.image.decode_jpeg(tf.read_file(filename))) # augmentation with Gaussian noise
    # return tf.image.rgb_to_grayscale(g_noise(tf.image.decode_jpeg(tf.read_file(filename)))) # augmentation with Gaussian noise
pass

def gazefan(x, reuse=True):
    # Input : 416 * 416
    with tf.variable_scope('gazefan', reuse=reuse):
        x = tf.image.resize_images(x, (256, 256))
        conv1 = tf.keras.layers.Conv2D(128, kernel_size=(11,11), strides=(2,2), activation=tf.nn.relu, padding="SAME")(x) #208
        conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7,7), strides=(1,1), activation=tf.nn.relu, padding="SAME")(conv1) #208
        conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7,7), strides=(1,1), activation=tf.nn.relu, padding="SAME")(conv1) #208
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(7,7), strides=(1,1), activation=tf.nn.relu, padding="SAME")(conv1) #208
        conv2 = tf.keras.layers.Conv2D(256, kernel_size=(7,7), strides=(2,2), activation=tf.nn.relu, padding="SAME")(conv1) #104
        conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5,5), strides=(1,1), activation=tf.nn.relu, padding="SAME")(conv2) #104
        conv2 = tf.keras.layers.Conv2D(32, kernel_size=(5,5), strides=(1,1), activation=tf.nn.relu, padding="SAME")(conv2) #104
        conv2 = tf.keras.layers.Conv2D(256, kernel_size=(5,5), strides=(1,1), activation=tf.nn.relu, padding="SAME")(conv2) #104
        conv3 = tf.keras.layers.Conv2D(128, kernel_size=(5,5), strides=(2,2), activation=tf.nn.relu, padding="SAME")(conv2) #52
        conv4 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation=tf.nn.relu, padding="SAME")(conv3) #26
        conv5 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation=tf.nn.relu, padding="SAME")(conv4) #13
        
        fin = tf.keras.layers.Flatten()(conv5)
        f1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(fin)
        f2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(f1)
        f3 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(f2)
        
        out = tf.keras.layers.Dense(2, activation=tf.nn.tanh)(f3)
    return out
pass 

def main():
    # training settings
    OUTPUT = 'rec.txt'
    outputf = open(OUTPUT, 'w', buffering=1)
    batch_size = 32
    training_steps = 3000000 # 100 epochs
    tr_dataset_folder = 'imgs-lefteyes'
    # tr_dataset_folder = 'imgs_s'

    tr_pics = glob.glob(tr_dataset_folder + '/*.jpg')
    # tr_jsons = glob.glob(tr_dataset_folder + '/*.json')
    tr_gaze_angles = np.array([get_look_vec_from_json(i.split('.')[0] + '.json') for i in tr_pics])
    
    # creating entry points
    pic_fns = tf.placeholder(tf.string,[None])
    gaze_angles = tf.placeholder(tf.float32,[None, 2])
    ts_pics = tf.placeholder(tf.uint8, [None, 600, 800, 3])
    
    # define the training dataset
    tr_dataset = tf.data.Dataset.from_tensor_slices({'img_fns':pic_fns, 'labs':gaze_angles})
    tr_dataset = tr_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
    tr_dataset = tr_dataset.prefetch(buffer_size=1000000)
    tr_dataset = tr_dataset.batch(batch_size)
    tr_dataset_iter = tr_dataset.make_initializable_iterator()
    tr_dataset_fetch = tr_dataset_iter.get_next()
    tr_imgs = tf.map_fn(lambda x: tf.cast(tf.reshape(get_gaze_pic(x), [600,800,3]), dtype=tf.float32) ,tr_dataset_fetch['img_fns'], dtype=tf.float32)
    tr_labs = tr_dataset_fetch['labs']
    
    # use gray level
    tr_imgs = tf.image.rgb_to_grayscale(tr_imgs)
    ts_pics = tf.image.rgb_to_grayscale(ts_pics)
    
    # augument with random crop and reside
    tr_imgs = tf.map_fn(lambda x: tf.image.resize_images(tf.image.random_crop(x, [(600 - tf.random.uniform([1])[0] * 200),(800 - tf.random.uniform([1])[0] * 200),1]), [600,800]), tr_imgs) 

    # define the computational graph
    # tr_preds = gazefan(tf.map_fn(lambda x: tf.image.per_image_standardization(x), tr_imgs, dtype=tf.float32), False)
    # ts_preds = gazefan(tf.map_fn(lambda x: tf.image.per_image_standardization(x), ts_pics, dtype=tf.float32), True)
    tr_preds = gazefan(tf.map_fn(lambda x: (x / 128) - 1, tr_imgs, dtype=tf.float32), False)
    ts_preds = gazefan(tf.map_fn(lambda x: (x / 128) - 1, ts_pics, dtype=tf.float32), True)
    

    # define loss and optimization
    tr_MSE = tf.reduce_mean(tf.pow((tr_preds - tr_labs), 2))
    # ts_MSE = tf.reduce_mean(tf.pow((ts_preds - tr_labs), 2))
    
    gazefan_var =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gazefan')
    # opt = tf.contrib.opt.AdamWOptimizer(1E-4, 1E-6).minimize(tr_MSE, var_list=gazefan_var)
    opt = tf.train.RMSPropOptimizer(1E-6, .5).minimize(tr_MSE, var_list=gazefan_var)
    
    ## training
    # setting trainer
    saver = tf.train.Saver(var_list=gazefan_var)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    # create training dataset
    sess.run(tr_dataset_iter.initializer, feed_dict={pic_fns: tr_pics, gaze_angles: tr_gaze_angles}) # dataset initializing
    
    # start training
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()
    for steps in range(training_steps):
        _, loss = sess.run([opt, tr_MSE])
        if steps % 500 == 0:
            saver.save(sess, 'models/gazefan', global_step=steps)
            print('steps:{}  loss:{}'.format(steps, loss))
            outputf.write('steps:{}  loss:{}\n'.format(steps, loss))
        pass
    pass

    sess.close() 
pass 

if __name__ == '__main__':
    main()
pass
