#!/usr/bin/python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc as sm
import tensorflow_datasets as tfds 
import jax 
from jax.experimental import optimizers
import numpy as np

global prngkey
prngkey = jax.random.PRNGKey(20) 

def G():
    pass 

def D():
    pass 

def 
    
def main():
    batch_size = 32
    enhance_G_sample_rate = 8/32
    output_g_sample_size = 5
    training_iter = 500000
    noize_dim = 1024
    alpha = 1. # constant for weaking the D
    softdec_c = 1e-1 # soft the one-hot
    # softdec_c = .0 # turn off soft the one-hot
    mbdl_sample_no = int( batch_size * (enhance_G_sample_rate + 1) ) * 3 # mini batch discrimination sample number
    mbdli_no = 20
    
    ## Import MNIST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    
    
    ## build GAN
    # announce the essential placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1]) # put the real samples
    Y = tf.placeholder(dtype=tf.float32, shape=[None,]) # the real and fake labels
    Z = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, noize_dim]) # the noise for generating samples
    # announce the weights of G and D
    [gw, dw] = W()
    
    # get samples from generator
    gX  = G(Z, gw, reuse=False)
    gXd = G(Z, gw, reuse=True, training=False)
    mbdl = minibatch_discrimonation(mbdl_sample_no, noize_dim, gw)
    mbdliG = minibatch_discrimonation_ind(mbdli_no, noize_dim, gw, gX)
    mbdliD = minibatch_discrimonation_ind(mbdli_no, noize_dim, gw, tf.concat([gX, X], axis=0))
    # mbdliDr = minibatch_discrimonation_ind(30, noize_dim, gw, X)
    # mbdliDf = minibatch_discrimonation_ind(30, noize_dim, gw, gX)
    # mbdliD = tf.concat([mbdliDf, mbdliDr], axis=0)
    # mbdliD = tf.concat([mbdliDf, tf.ones_like(mbdliDr)], axis=0)
    
    # define the loss and optimizers, real samples are labeled 1 and fake samples are labeled 0
    logits_4Df = D(gX, dw, reuse=False) #label: (fake, real)
    logits_4Dr = D( X, dw)
    logits_4D  = tf.concat([logits_4Df, logits_4Dr], axis=0)
    logits_4G = D(gX, dw)
    
    ## loss definition
    # focal loss, using the minibatch distrinimation result as focal. The weighs with large resutls are bad and will get large gradients.
    # loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits_4D) * tf.pow(tf.exp(-mbdl), 1))  
    # loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits_4D) * tf.pow(tf.exp(-mbdliD), 2))
    loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits_4D) )
    # loss_D = tf.reduce_mean(tf.losses.hinge_loss(labels=Y, logits=logits_4D, reduction=tf.losses.Reduction.NONE) )
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G) * tf.pow(tf.exp(mbdl), 1)) 
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G) * tf.pow(tf.exp(mbdliG), 2))
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G) )
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G) )
    loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4Df), logits=logits_4Df) )
    # loss_G = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_4G), logits=logits_4G) )
    # loss_G = -tf.reduce_min(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_4G), logits=logits_4G) )
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_4G), logits=logits_4G) )#https://github.com/soumith/ganhacks # 2-flip label
    # regulize the loss 
    # loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_4G), logits=logits_4G)) + 0.01 * mbdl
    
    ## optimizer definition
    
    # Since the defalut setting of Tensorflow will update every variable during each optimizor, we need to lock the 
    # variables which we don't want it be influenced when runing optimization
    G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
    # G_var = [var for var in tf.trainable_variables() if var.name.startswith("Generator")]
    # D_var = [var for var in tf.trainable_variables() if var.name.startswith("Discriminator")]
    
    
    # opt_D = tf.train.AdamOptimizer(1e-5, beta1=0.618).minimize(loss_D)
    # opt_G = tf.train.AdamOptimizer(1e-5, beta1=0.618).minimize(loss_G)
    # opt_D = tf.train.AdamOptimizer(1e-3).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.AdamOptimizer(1e-3).minimize(loss_G, var_list = G_var)# opt_D = tf.train.AdamOptimizer(1e-5, beta1=0.382).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.AdamOptimizer(1e-4, beta1=0.618).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.AdamOptimizer(1e-4, beta1=0.618).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.AdamOptimizer(1e-4, beta1=0.618).minimize(loss_G)
    opt_D = tf.train.AdamOptimizer(1e-5, beta1=0.382).minimize(loss_D, var_list = D_var)
    opt_G = tf.train.AdamOptimizer(1e-5, beta1=0.618).minimize(loss_G, var_list = G_var) 
    # opt_D = tf.train.AdamOptimizer(1e-4, beta1=0.382).minimize(loss_D)
    # opt_G = tf.train.AdamOptimizer(1e-3).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.RMSPropOptimizer(1e-5, momentum=0.382).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.RMSPropOptimizer(1e-5, momentum=0.382).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_G, var_list = G_var)
    # opt_G = tf.train.AdadeltaOptimizer(1e-4).minimize(loss_G, var_list = G_var)
    # opt_D = tf.train.AdadeltaOptimizer(1e-6).minimize(loss_D, var_list = D_var)
    # opt_D = tf.train.AdagradOptimizer(1e-4).minimize(loss_D, var_list = D_var)
    # opt_G = tf.train.AdagradOptimizer(1e-4).minimize(loss_G, var_list = G_var)
    # opt_D = tf.contrib.opt.NadamOptimizer(1e-5, beta1=0.382).minimize(loss_D, var_list = D_var)
    # opt_G = tf.contrib.opt.NadamOptimizer(1e-5, beta1=0.382).minimize(loss_G, var_list = G_var)


    ## start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    with tf.Session(config=config) as sess:
        feed_in_G_sample_size = int(batch_size * enhance_G_sample_rate)
        
        # y = np.hstack([np.zeros([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real)
    
        sess.run(tf.global_variables_initializer())
        for iter in range(training_iter):
            # softdec = softdec_c * np.random.random()
            # softdec = softdec_c # without noise
            
            # y = np.hstack([np.zeros([feed_in_G_sample_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            
            # x, _ = mnist.train.next_batch(batch_size)
            # # x = mnist_select(mnist, int(iter/300) % 9, batch_size) # this will enforce the number class in to the dataset
            # x = x.reshape([-1, 28, 28, 1])
            # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            
            # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            # # z = np.random.uniform(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            
            ## strategy 1: update simultaneously
            #####
            ## D should be trained using equal number of fake and real samples.
            ## More training samples are also feeded into D than G when training.
            ## During optimize G, if G generate less samples, the gradients would be more specific to G, 
            ## and this make G can understand more about which parts should be modified.
            ## So the strategy here uses few generated sample, but more real sample to D for getting the gradients.
            #####
            # # train D
            # x, _ = mnist.train.next_batch(batch_size)
            # x = x.reshape([-1, 28, 28, 1])
            # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            # # y = np.hstack([np.zeros([batch_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            # y = np.hstack([np.zeros([batch_size,]), np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            # #z = np.random.normal(size = batch_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            # # since 0~9 should be sampled equally, the final channel are set the uniform rather than normaldistribution
            # z = np.random.normal(size = batch_size * 1 * 1 * (noize_dim-1)).reshape([-1, 1, 1, noize_dim-1])
            # z = np.concatenate((z, np.random.uniform(size=[batch_size, 1, 1, 1]) * 2 - 1), axis=-1)
            # # z = np.concatenate((z, np.random.randint(10, size=[batch_size, 1, 1, 1]) * .2 - 1), axis=-1)
            
            
            # # _, CmbdliD = sess.run([opt_D, mbdliD], feed_dict={X:x, Y:y, Z:z}) 
            # _ = sess.run([opt_D], feed_dict={X:x, Y:y, Z:z})
            
            # # train G
            # x, _ = mnist.train.next_batch(batch_size)
            # x = x.reshape([-1, 28, 28, 1])
            # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            # # y = np.hstack([np.zeros([feed_in_G_sample_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            # y = np.hstack([np.zeros([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real), soft one-hots
            # # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * (noize_dim-1)).reshape([-1, 1, 1, noize_dim-1])
            # z = np.concatenate((z, np.random.uniform(size=[feed_in_G_sample_size, 1, 1, 1]) * 2 - 1), axis=-1)
            # # z = np.concatenate((z, np.random.randint(10, size=[feed_in_G_sample_size, 1, 1, 1]) * .2 - 1), axis=-1)
            # # _, CmbdliG = sess.run([opt_G, mbdliG], feed_dict={X:x, Y:y, Z:z}) 
            # _ = sess.run([opt_G], feed_dict={Z:z})
            
            # # get the output
            # cX, Closs_D, Closs_G, Cmbdl = sess.run([gX, loss_D, loss_G, mbdl], feed_dict={X:x, Y:y, Z:z}) 
            
            ## strategy 2: make D stroger but update less times
            # # train D
            # if iter % 100 == 0 : 
                # for D_iter in range(20):
                    # x, _ = mnist.train.next_batch(batch_size)
                    # x = x.reshape([-1, 28, 28, 1])
                    # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
                    # z = np.random.normal(size = batch_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                    # y = np.hstack([np.zeros([batch_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
                    # # _, CmbdliD = sess.run([opt_D, mbdliD], feed_dict={X:x, Y:y, Z:z}) 
                    # _ = sess.run([opt_D], feed_dict={X:x, Y:y, Z:z})
            
            # # train G
            # x, _ = mnist.train.next_batch(batch_size)
            # x = x.reshape([-1, 28, 28, 1])
            # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            # y = np.hstack([np.zeros([feed_in_G_sample_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
            # # _, CmbdliG = sess.run([opt_G, mbdliG], feed_dict={Z:z}) 
            # _ = sess.run([opt_G], feed_dict={Z:z}) 
            
            # # get the output
            # # cX, Closs_D, Closs_G, Cmbdl = sess.run([gX, loss_D, loss_G, mbdl], feed_dict={X:x, Y:y, Z:z}) 
            # cX, Closs_D, Closs_G = sess.run([gX, loss_D, loss_G], feed_dict={X:x, Y:y, Z:z}) 
            
            ## strategy 3: update G more time (because G is hard to train) and fix D
            # train D
            for Di in range(1):
                softdec = softdec_c * np.random.random()
                generated_sample_size_for_D = int(batch_size * 1.)
                x, _ = mnist.train.next_batch(batch_size)
                x = x.reshape([-1, 28, 28, 1])
                x = (x - 0.5)/ 0.5 # normalize from -1 to 1
                # y = np.hstack([np.zeros([batch_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
                y = np.hstack([np.zeros([generated_sample_size_for_D,]), np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
                z = np.random.normal(size = generated_sample_size_for_D * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                # since 0~9 should be sampled equally, the final channel are set the uniform rather than normaldistribution
                # z = np.random.normal(size = generated_sample_size_for_D * 1 * 1 * (noize_dim-1)).reshape([-1, 1, 1, noize_dim-1])
                # z = np.concatenate((z, np.random.uniform(size=[generated_sample_size_for_D, 1, 1, 1]) * 2 - 1), axis=-1)
                # z = np.concatenate((z, np.random.randint(10, size=[batch_size, 1, 1, 1]) * .2 - 1), axis=-1)
                
                # _, CmbdliD = sess.run([opt_D, mbdliD], feed_dict={X:x, Y:y, Z:z}) 
                _ = sess.run([opt_D], feed_dict={X:x, Y:y, Z:z})
            
            # train G
            x, _ = mnist.train.next_batch(batch_size)
            x = x.reshape([-1, 28, 28, 1])
            x = (x - 0.5)/ 0.5 # normalize from -1 to 1
            # y = np.hstack([np.zeros([feed_in_G_sample_size,]) + softdec, np.ones([batch_size,]) - softdec]) #(fake, real), soft one-hots
            y = np.hstack([np.zeros([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real), soft one-hots
            for Gi in range(5):
                z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * (noize_dim-1)).reshape([-1, 1, 1, noize_dim-1])
                # z = np.concatenate((z, np.random.uniform(size=[feed_in_G_sample_size, 1, 1, 1]) * 2 - 1), axis=-1)
                # z = np.concatenate((z, np.random.randint(10, size=[feed_in_G_sample_size, 1, 1, 1]) * .2 - 1), axis=-1)
                # _, CmbdliG = sess.run([opt_G, mbdliG], feed_dict={X:x, Y:y, Z:z}) 
                _ = sess.run([opt_G], feed_dict={Z:z})
            
            # get the output
            # cX, Closs_D, Closs_G, Cmbdl = sess.run([gX, loss_D, loss_G, mbdl], feed_dict={X:x, Y:y, Z:z}) 
            Closs_D, Closs_G, _ = sess.run([loss_D, loss_G, opt_G], feed_dict={X:x, Y:y, Z:z})
            
            ## strategy 4: fool the D
            # if np.random.random() > .98:
                # x, _ = mnist.train.next_batch(batch_size)
                # x = x.reshape([-1, 28, 28, 1])
                # x = (x - 0.5)/ 0.5 # normalize from -1 to 1
                # z = np.random.normal(size = feed_in_G_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                # y_f = np.hstack([np.ones([feed_in_G_sample_size,]), np.ones([batch_size,])]) #(fake, real)
                # sess.run(opt_D, feed_dict={X:x, Y:y_f, Z:z}) 
            
            if iter % 500 == 0 :
                # print('iteration:{} loss_D:{} loss_G:{} mbdliD:{} mbdliG:{} mbdl:{}'.format(iter, Closs_D, Closs_G, CmbdliD, CmbdliG, Cmbdl))
                print('iteration:{} loss_D:{} loss_G:{}'.format(iter, Closs_D, Closs_G))
                    
                for gi in range(output_g_sample_size):
                    z = np.random.normal(size = output_g_sample_size * 1 * 1 * noize_dim).reshape([-1, 1, 1, noize_dim])
                    # z = np.random.normal(size = 1 * 1 * 1 * (noize_dim-1)).reshape([-1, 1, 1, noize_dim-1])
                    # z = np.concatenate((z, np.random.uniform(size=[1, 1, 1, 1]) * 2 - 1), axis=-1)
                    # z = np.concatenate((z, np.random.randint(10, size=[1, 1, 1, 1]) * .2 - 1), axis=-1)
                    cX = sess.run(gX, feed_dict={Z:z})
                    visg = (cX[0].T + 1)/2
                    # visr = (x[0].T + 1)/2
                    visgp = np.vstack([visg, visg, visg]).T
                    visgp *= 255
                    visgp = sess.run(tf.image.resize_images(visgp,[64,64]))
                    # visrp = np.vstack([visr, visr, visr]).T
                    # visrp *= 255
                    visgp.astype(np.uint8)
                    sm.imsave('G/g{}_{}.jpg'.format(gi, iter),visgp)
                    # visrp.astype(np.uint8)
                    # sm.imsave('G/r{}_{}.jpg'.format(gi, iter),visrp)
            
    
pass
    
    
    
if __name__=='__main__':
    main()
pass

