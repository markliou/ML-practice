import numpy as np 
import tensorflow as tf 

def Attention(Q, K, name='att'):
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
    attention_map = tf.nn.softmax(attention_map, axis=1) # consider only the keys for attention
    attention = tf.reshape(Kf,[tf.shape(Kf)[0], tf.shape(Kf)[1], 1]) * attention_map 
    attention = tf.reduce_sum(attention, axis=1)
    gamma = tf.get_variable(name+"att_gamma", [1], initializer=tf.constant_initializer(0.0)) # set the gamma as learnable variable
    return tf.reshape(attention, tf.shape(Q)) * tf.nn.sigmoid(gamma) + Q * tf.nn.sigmoid(1 - gamma) # V
pass 

def self_spatial_attention(x, compression_channel_no = 16):
    '''
    This section is implemented according to SAGAN (https://arxiv.org/abs/1805.08318).
    The source code can also be found at http://www.twistedwg.com/2018/06/27/SAGAN-code.html .

    '''
    # For f and g:
    # To reduce the computing resource, the input can be compressed.
    # We only need to focus on the shape of attention map equal to the number of each spatial map.
    # So, please feel free to set the compression channel number. 
    f = tf.nn.relu(tf.layers.conv2d(x, compression_channel_no, [1,1], [1,1], "SAME"))
    g = tf.nn.relu(tf.layers.conv2d(x, compression_channel_no, [1,1], [1,1], "SAME"))
    # For h:
    # the shape should be equal to input x
    h = tf.nn.relu(tf.layers.conv2d(x, x.shape[3].value, [1,1], [1,1], "SAME"))
    s = tf.matmul(tf.reshape(g, [tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], compression_channel_no]), tf.reshape(f, [tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], compression_channel_no]), transpose_b=True)  # [bs, N, N]
    beta = tf.nn.softmax(s, axis=-1)  # attention map
    o = tf.matmul(beta,tf.reshape(h, [tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], tf.shape(x)[3]]))  # [bs, N, C]
    o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0)) # set the gamma as learnable variable
    x = o + gamma * x
    return x
pass

def main():

    # use difference source and target 
    S = tf.zeros([10, 28, 28, 3])
    T = tf.zeros([10, 14, 14, 1024])
    A = Attention(T, S, 'att1')
    print(A) # output should have the same shape of T (10, 14, 14, 1024)

    # puting self as two inputs will be self-attention
    A = Attention(T, T, 'att2')
    print(A)

    # the attention from SAGAN
    A = self_spatial_attention(T)
    print(A)

pass

if __name__ == '__main__':
    main()
pass