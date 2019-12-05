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
    output_seq_length = 30
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
        en_hiden_stats, encoder_res = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True, return_state=True)(inputs)
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
            de_hiden_stats_Q = tf.reshape(de_hiden_stats_Q, [None])
            
            # calculate the dot scores between encoder_res_K and de_hiden_stats_Q. Get the softmax of each state
            scores = tf.map_fn(lambda x: tf.reduce_sum( x * de_hiden_stats_Q), encoder_res_K)
            weights = tf.nn.softmax(scores)

            # using the K as the final output value directly according to the attention weights
            V = tf.reduce_sum(encoder_res_K * tf.expand_dims(weights, 0), axis=0)

            return V, weights
        pass 

        ##### decoder
        en_hiden_stats, encoder_res = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True, return_state=True)(inputs)
        # print(encoder_res)
        # print(en_hiden_stats)
        # exit()

        #encoder_outputs은 Encoder의 output이다. 보통 Memory라 불린다. 여기서는 toy model이기 때문에 ranodm값을 생성하여 넣어 준다.
        encoder_outputs = tf.convert_to_tensor(np.random.normal(0,1,[batch_size,20,30]).astype(np.float32)) # 20: encoder sequence length, 30: encoder hidden dim
        
        # encoder_outpus의 길이는 20이지만, 다음과 같이 조절할 수 있다.
        input_lengths = [5,10,20]  # encoder에 padding 같은 것이 있을 경우, attention을 주지 않기 위해
        
        # attention mechanism  # num_units = Na = 11
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths,normalize=False)

        
        attention_initial_state = cell.zero_state(batch_size, tf.float32)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=13,initial_cell_state=attention_initial_state,
                                                   alignment_history=alignment_history_flag,output_attention=True)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_dim)
        
        # 여기서 zero_state를 부르면, 위의 attentionwrapper에서 넘겨준 attention_initial_state를 가져온다. 즉, AttentionWrapperState.cell_state에는 넣어준 값이 들어있다.
        initial_state = cell.zero_state(batch_size, tf.float32) # AttentionWrapperState
 
        helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state)    

        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True)
     
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
     
        opt = tf.train.AdamOptimizer(0.01).minimize(loss)
        
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            loss_,_ =sess.run([loss,opt])
            print("{} loss: = {}".format(i,loss_))
        
        if alignment_history_flag ==False:
            print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,o2) #batch_size, seq_length, outputs
     
        print("\n\nlast_state: ",last_state)
        if alignment_history_flag == False:
            print(sess.run(last_state)) # batch_size, hidden_dim
        else:
            print("alignment_history: ", last_state.alignment_history.stack())
            alignment_history_ = sess.run(last_state.alignment_history.stack())
            print(alignment_history_)
            print("alignment_history sum: ",np.sum(alignment_history_,axis=-1))
            
            print("cell_state: ", sess.run(last_state.cell_state))
            print("attention: ", sess.run(last_state.attention))
            print("time: ", sess.run(last_state.time))
            
            alignments_ = sess.run(last_state.alignments)
            print("alignments: ", alignments_)
            print('alignments sum: ', np.sum(alignments_,axis=1))   # alignments의 합이 1인지 확인
            print("attention_state: ", sess.run(last_state.attention_state))

        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
     
        p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)
        print("loss: {:20.6f}".format(sess.run(loss)))
        print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )   

if __name__ == '__main__':
    attention_test()