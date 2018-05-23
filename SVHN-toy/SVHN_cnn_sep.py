#!/usr/bin/python3
#####
## MIT license
## 20180520 markliou
#####

import cv2
import tensorflow as tf
import numpy as np
# import tools


def stack_dataset(data_list, data):
    # return the NHWC format, HW=32*32
    size = 32
    # data_container = np.stack( [cv2.resize(cv2.imread(data[i]['addr']), (size,size)) for i in data_list] )
    data_container = np.stack( [ (cv2.resize(cv2.imread(data[i]['addr']), (size,size))-128)/256 for i in data_list] )
    return data_container
    

def stack_labels(data_list, data):
    digit_no = 6
    num_class = 10
    oc_container = []
    
    for i in data_list:
        oc_container_t = []
        # if the labels occupy than expected
        if len(data[i]['label']) > digit_no: 
            print('digital position is not enought')
            exit()
        
        # rearrange the digital number
        uni_label = data[i]['label']
        while len(uni_label) < digit_no:
            uni_label = '0' + uni_label
    
        for j in range(digit_no):
            oh = np.zeros([num_class,])
            oh[ int(uni_label[j]) ] = 1
            oc_container_t.append(oh)
        
        oc_container.append(np.stack(oc_container_t))
        
    # label_container = np.stack( [ data[i]['label'] for i in data_list] )
    label_container = np.stack( oc_container )
    # print(label_container)
    return label_container
    
def cnn(X):
    # initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.keras.initializers.he_normal()
    # initializer = tf.random_normal
    # initializer = tf.zeros
    num_class = 10 #0~9
    kernels = {
        'inw1': tf.Variable(initializer([1, 1, 3, 1])),
        'inw2': tf.Variable(initializer([1, 1, 1, 3])),
        'wc1': tf.Variable(initializer([7, 7, 3, 512])),
        'wc2': tf.Variable(initializer([5, 5, 512, 256])),
        'wc3': tf.Variable(initializer([5, 5, 256, 512])),
        'wc4': tf.Variable(initializer([3, 3, 512, 256])),
        'wc5': tf.Variable(initializer([3, 3, 256, 512])),
    }
    fc = {
        'wf1': tf.Variable(initializer([1*1*512, 1024])),
        'wf2': tf.Variable(initializer([1024,512])),
        'out1': tf.Variable(initializer([512,num_class])),
        'out2': tf.Variable(initializer([512,num_class])),
        'out3': tf.Variable(initializer([512,num_class])),
        'out4': tf.Variable(initializer([512,num_class])),
        'out5': tf.Variable(initializer([512,num_class])),
        'out6': tf.Variable(initializer([512,num_class])),
    }
    bias = {
        'cb1': tf.Variable(initializer([512])),
        'cb2': tf.Variable(initializer([256])),
        'cb3': tf.Variable(initializer([512])),
        'cb4': tf.Variable(initializer([256])),
        'cb5': tf.Variable(initializer([512])),
        'fb1': tf.Variable(initializer([1024])),
        'fb2': tf.Variable(initializer([512])),
        'out1': tf.Variable(initializer([num_class])),
        'out2': tf.Variable(initializer([num_class])),
        'out3': tf.Variable(initializer([num_class])),
        'out4': tf.Variable(initializer([num_class])),
        'out5': tf.Variable(initializer([num_class])),
        'out6': tf.Variable(initializer([num_class])),
    }
    
    # in1 = tf.nn.relu(tf.nn.conv2d(X, kernels['inw1'], strides=[1, 1, 1, 1], padding='SAME')) # 32
    # in2 = tf.nn.relu(tf.nn.conv2d(in1, kernels['inw2'], strides=[1, 1, 1, 1], padding='SAME')) # 32
    
    cl1 = tf.nn.relu(tf.nn.conv2d(X, kernels['wc1'], strides=[1, 2, 2, 1], padding='SAME') + bias['cb1'])
    # cl1 = tf.nn.relu(tf.nn.conv2d(in2, kernels['wc1'], strides=[1, 2, 2, 1], padding='SAME') + bias['cb1'])# 16
    # cl1 = tf.layers.batch_normalization(cl1)
    cl2 = tf.nn.relu(tf.nn.conv2d(cl1, kernels['wc2'], strides=[1, 2, 2, 1], padding='SAME') + bias['cb2']) # 8
    # cl2 = tf.layers.batch_normalization(cl2)
    cl3 = tf.nn.relu(tf.nn.conv2d(cl2, kernels['wc3'], strides=[1, 2, 2, 1], padding='SAME') + bias['cb3']) # 4
    # cl3 = tf.layers.batch_normalization(cl3)
    cl4 = tf.nn.relu(tf.nn.conv2d(cl3, kernels['wc4'], strides=[1, 2, 2, 1], padding='SAME') + bias['cb4']) # 2
    # cl4 = tf.layers.batch_normalization(cl4)
    cl5 = tf.nn.relu(tf.nn.conv2d(cl4, kernels['wc5'], strides=[1, 2, 2, 1], padding='SAME') + bias['cb5']) # 1
    # cl5 = tf.layers.batch_normalization(cl5)
    
    ff = tf.reshape(cl5, [-1, 1*1*512 ])
    # ff = tf.reshape(cl5, [-1, fc['wf1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.matmul(ff,fc['wf1']) + bias['fb1'])
    # fc1 = tf.layers.batch_normalization(fc1)
    fc2 = tf.nn.relu(tf.matmul(fc1,fc['wf2']) + bias['fb2'])
    # fc2 = tf.layers.batch_normalization(fc2)
    
    logit1 = (tf.matmul(fc2, fc['out1']) + bias['out1'])
    logit2 = (tf.matmul(fc2, fc['out2']) + bias['out2'])
    logit3 = (tf.matmul(fc2, fc['out3']) + bias['out3'])
    logit4 = (tf.matmul(fc2, fc['out4']) + bias['out4'])
    logit5 = (tf.matmul(fc2, fc['out5']) + bias['out5'])
    logit6 = (tf.matmul(fc2, fc['out6']) + bias['out6'])
    
    return(tf.stack([logit1, logit2, logit3, logit4, logit5, logit6], axis=1))
    # return logit1

def subdataset_generator(idx):
    idx_s = []
    while(1):
        if len(idx_s) == 0:
            idx_s = idx.copy()
            np.random.shuffle(idx_s)
        yield idx_s.pop()
    

def main():
    digit_no = 6
    
    ## build CNN
    opt_iteration = 10000000
    batch_size = 32
    num_class = 10 #0~9
    learning_rate = 1E-5
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y = tf.placeholder(tf.float32, [None, 6, num_class])
    logits = cnn(X)
    
    # jointly training 
    CE = tf.reduce_sum(tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits , labels=Y, dim=-1), axis=-1 ))
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(CE)
    # opt = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate).minimize(CE)
    
    # alternative training
    CEA, optA = [], []
    for digit_i in range(digit_no):
        CEA.append( tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits[digit_i] , labels=Y[digit_i])
                  ) )
        optA.append( tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(CEA[digit_i]) )
    
    ## setting the dataset paramters
    # training_data_list = 'dataset/train_t.list'
    training_data_list = 'dataset/train.list'
    # training_labels = 'dataset/train_t.labels'
    training_labels = 'dataset/train.labels'
    
    data = {} # format : {filename:{label,addr}}
    
    # readling the labels
    FH_label = open(training_labels,'r')
    for line in FH_label.readlines():
        element = line.rstrip().split(',')
        if len(element) == 0 : # avoiding the final blank
            continue
        data[element[0]] = {'label':element[2]}
    FH_label.close()
    
    # readling the training pics 
    FH_list = open(training_data_list,'r')
    for line in FH_list.readlines():
        data_add = line.rstrip()
        if data_add == '' : # avoiding the final blank
            continue
        data[data_add.split('/')[-1]]['addr'] = data_add
    FH_list.close()
    
    # print(data)
    subdataset_g = subdataset_generator(list(data.keys()))
    
    ## start to run cnn
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer()) # initializer the computational graph
        saver = tf.train.Saver() # creating the saver object for saving models
        saver.save(sess, 'models/SVHN.ckpt', global_step=5000) # setting the saver. The model will be saved every 1000 steps automatically
        
        # construct the feed in dataset
        # data_container = stack_dataset(list(data.keys()), data)
        # label_container = stack_labels(list(data.keys()), data)
        # data_list = [ next(subdataset_g) for idxi in range(batch_size)]
        # data_container = stack_dataset(data_list, data)
        # label_container = stack_labels(data_list, data)
        
        
        # loss = sess.run(CE, feed_dict={X:data_container , Y:label_container})
        # c_logits = sess.run(logits, feed_dict={X:data_container[0:1] , Y:label_container})

        for step in range(opt_iteration):
            data_list = [ next(subdataset_g) for idxi in range(batch_size)]
            data_container = stack_dataset(data_list, data)
            label_container = stack_labels(data_list, data)
            
            ## alternative training
            for digit_i in range(digit_no):
                sess.run(optA[digit_i], feed_dict={X:data_container , Y:label_container})
            [loss] = sess.run([CE], feed_dict={X:data_container , Y:label_container})
            
            ## jointly training
            # [_, loss] = sess.run([opt, CE], feed_dict={X:data_container , Y:label_container}) 
            
            if step%5 == 0 :
                print('step:{} loss:{}'.format(step,loss))
                # show some results
                data_list = [ next(subdataset_g) for idxi in range(2)]
                data_container = stack_dataset(data_list, data)
                label_container = stack_labels(data_list, data)
                [pred,ans] = sess.run([[tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)],[tf.argmax(tf.nn.softmax(label_container, axis=-1), axis=-1)]], feed_dict={X:data_container , Y:label_container})
                t_logits = sess.run(tf.nn.softmax(logits), feed_dict={X:data_container , Y:label_container})
                print('{}   \n{}'.format(data_list,t_logits))
                print('{}   \npred:{} \nans :{}'.format(data_list,pred,ans))
                # print('{}   \n{}\n{}'.format(data_list,t_logits,np.array(t_logits.shape)))
    
        #### test section
        data_list = [ next(subdataset_g) for idxi in range(5)]
        data_container = stack_dataset(data_list, data)
        label_container = stack_labels(data_list, data)
        t_logits = sess.run(tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1), feed_dict={X:data_container , Y:label_container})
        # print(np.array(t_logits).shape)
        print('{}   \n{}'.format(data_list,t_logits))
    



if __name__=='__main__':
    main()