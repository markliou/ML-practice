import numpy as np 
import tensorflow as tf 

def Attention(Q, K):
    '''
    Use simple dot product to get the attention. The variable names and 
    concepts are still deviative from "Attention is all you need".
    (https://arxiv.org/abs/1706.03762)
    The main idea of attention for convolution is from SAGAN 
    (https://arxiv.org/abs/1805.08318)
    Q and K have format of NHWC.
    '''
    Qf, Kf = tf.layers.flatten(Q), tf.layers.flatten(K)
    attention_map = tf.reshape(Kf,[tf.shape(Kf)[0], tf.shape(Kf)[1], 1]) * tf.reshape(Qf,[tf.shape(Qf)[0], 1, tf.shape(Qf)[1]]) 
    attention_map_shape = tf.shape(attention_map)
    attention_map = tf.reshape(tf.nn.softmax(tf.layers.flatten(attention_map)), attention_map_shape) # whole map as attention
    # attention_map = tf.nn.softmax(attention_map, axis=1) # consider only the keys for attention
    attention = tf.reshape(Kf,[tf.shape(Kf)[0], tf.shape(Kf)[1], 1]) * attention_map 
    attention = tf.reduce_sum(attention, axis=1)
    return tf.reshape(attention, tf.shape(Q)) # V
pass 

def main():

    # use difference source and target 
    S = tf.zeros([10, 28, 28, 3])
    T = tf.zeros([10, 14, 14, 1024])
    A = Attention(T, S)
    print(A) # output should have the same shape of T (10, 14, 14, 1024)

    # puting self as two inputs will be self-attention
    A = Attention(T, T)
    print(A)

pass

if __name__ == '__main__':
    main()
pass