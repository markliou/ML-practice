# this code aim to study how to apply the attention mechanism 
# using the tensorflow RNN cell. 
# The original source is: 
# https://github.com/hccho2/Tensorflow-RNN-Tutorial/tree/master/4.%20Attention%20with%20Tensorflow

import tensorflow as tf
import numpy as np

tf.reset_default_graph()
def attention_test():
    vocab_size = 5
    SOS_token = 0
    EOS_token = 4
    
    x_data = np.array([[SOS_token, 3, 1, 2, 3, 2],[SOS_token, 3, 1, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[1,2,0,3,2,EOS_token],[3,2,3,3,1,EOS_token],[3,1,1,2,0,EOS_token]],dtype=np.int32)
    Y = tf.convert_to_tensor(y_data)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    encoder_hidden_dim = 13
    decoder_hidden_dim = 17
    attenion_embeding_dim = 11
    seq_length = x_data.shape[1]
    output_seq_length = 50
    embedding_dim = 8

    # init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    # print(init)
    # exit()
    
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        ##### preprocessing the sequences
        # padding the sequences
        padding_pre = tf.keras.preprocessing.sequence.pad_sequences(x_data)
        # embedding the sequences
        inputs = tf.keras.layers.Embedding(input_dim=25, output_dim=embedding_dim, mask_zero=True)(padding_pre) # shape:[3,6,8]
        
        ##### encoder
        # the encoder will give the shape of [3,6,13] of the states, and the final output of [3,13]
        en_hiden_stats, encoder_res = tf.keras.layers.GRU(units=encoder_hidden_dim, return_sequences=True, return_state=True)(inputs)
        # print(encoder_res)
        # print(en_hiden_stats)
        # exit()

        ##### attention machanism
        def attention(decoder_res, en_hiden_stats): # give one sample per time would be easier to implement
            ## we use the linear transforming to handle different shape tensors, that we can calcualted the similarity
            
            # the en_hiden_state are expected to be the shape of [6,13]. Hence using a single dense layer can handle the problem
            encoder_res_K = tf.keras.layers.Dense(attenion_embeding_dim)(en_hiden_stats) # shape:[6, 11]
            
            # the decoder_res is expect to be the shape of [17]. To simply using the dense layer, expanding the dimention would be an easy way
            de_hiden_stats_Q = tf.expand_dims(decoder_res, 0) # shape:[1, 17]
            de_hiden_stats_Q = tf.keras.layers.Dense(attenion_embeding_dim)(de_hiden_stats_Q) # shape:[1, 11]
            de_hiden_stats_Q = tf.reshape(de_hiden_stats_Q, [-1])
            # exit()
            
            # calculate the dot scores between encoder_res_K and de_hiden_stats_Q. Get the softmax of each state
            scores = tf.map_fn(lambda x: tf.reduce_sum( x * de_hiden_stats_Q), encoder_res_K )
            weights = tf.nn.softmax(scores)
            # print(scores)
            # print(weights)
            # exit()

            # using the K as the final output value directly according to the attention weights. Final V should be [13]
            V = en_hiden_stats * tf.expand_dims(weights, 1)
            # print(V)
            # exit()
            V = tf.reduce_sum(V, axis=0)
            # print(V)
            # exit()

            return V, weights
            # return V
        pass 

        ##### decoder
        # Feeding the final state of the encoder into the decoder
        # The shape of output state of the decoder is [3, 17]. If the decoder state is concated with attention, the embedding 
        # is expected to be [30] (17+13)
        decoder = tf.keras.layers.GRU(units=decoder_hidden_dim, return_sequences=False, return_state=False)
        
        # The input of the decoder is expected to be the embedding shape of [30]. So, the inputs from the final output of encoder,
        # which is [13] are padded using 0.
        padded_encoder_res = tf.map_fn(lambda x: tf.concat([x, tf.zeros([17], dtype=tf.float32)], axis=-1), encoder_res)
        # print(encoder_res)
        # print(padded_encoder_res)
        # exit()

        # get the first state for get the attention. Since the RNN cell have the sequence length in the Rank 3 tensor, 
        # the state from previous RNN would need to expand the dimention
        padded_encoder_res = tf.expand_dims(padded_encoder_res, 1) # shape: [3, 1, 30]
        de_state = decoder(padded_encoder_res) # shape: [3,17]
        # print(de_state)
        # exit()

        ## since the attention machanism is implement with unstack form, the samples are 
        ## unstack here. But this style is not good.
        en_hiden_stats_slices = tf.unstack(en_hiden_stats, axis=0)
        # print(en_hiden_stats_slices)
        de_state_slices = tf.unstack(de_state, axis=0)
        # print(de_state_slices)
        # exit()
        batch_collector = []
        for c_batch_ind in range(len(de_state_slices)):
            seq_collector = []
            de_state_slice = de_state_slices[c_batch_ind]
            en_hiden_stats_slice = en_hiden_stats_slices[c_batch_ind]
            for c_output_seq_len in range(output_seq_length - 1):
                ### get the attention
                V, weights = attention(de_state_slice, en_hiden_stats_slice)
                ### merge the attention into the input
                de_input = tf.concat([de_state_slice, V], axis=-1) # shape: [30]
                ### decoding
                decoder_res = tf.expand_dims(tf.expand_dims(de_input, 0), 0) # shape: [1, 1, 30]
                de_state_slice = decoder(decoder_res, initial_state=tf.expand_dims(de_state_slice, 0)) # shape: [1,17]
                # de_state_slice = decoder(decoder_res, initial_state=None) # shape: [1,17]
                de_state_slice = tf.reshape(de_state_slice, [-1]) # shape: [17]
                ### recording the states
                seq_collector.append(tf.identity(de_state_slice))
            pass
            # print(seq_collector)
            # exit()
            batch_collector.append(tf.stack(seq_collector, axis=0))
            
        pass
        # print(batch_collector)
        # exit()
        
        decoder_output = tf.stack(batch_collector, axis=0)
        print(decoder_output)

    sess.run(tf.global_variables_initializer())
    return sess.run(decoder_output)





if __name__ == '__main__':
    output = attention_test()
    print(output)
    print(output.shape)