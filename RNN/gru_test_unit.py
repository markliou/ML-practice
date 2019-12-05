import tensorflow as tf 
import numpy as np 



def main():
    print(tf.__version__)

    a = tf.zeros([3,30,125])
    
    a1 = [
          [i for i in range(np.random.randint(20))], 
          [i for i in range(np.random.randint(20))],
          [i for i in range(np.random.randint(20))],
          [i for i in range(np.random.randint(20))],
          [i for i in range(np.random.randint(20))]
         ] 
    print(a1)
    
    print("\n=== padding ===")
    padding_a1_pre = tf.keras.preprocessing.sequence.pad_sequences(a1)
    padding_a1_pos = tf.keras.preprocessing.sequence.pad_sequences(a1, padding='post')
    print(padding_a1_pre)
    print('---')
    print(padding_a1_pos)

    print("\n=== masking ===")
    embedding = tf.keras.layers.Embedding(input_dim=25, output_dim=20, mask_zero=True)(padding_a1_pos)
    ## note ##
    # The embedding will change one-hot to the target dimentions. Keras emdedding layer is feeded the index array, and will give the 
    # final embedding results. This action is quite different from the "Dense layer"
    ##########
    embedding_mask = embedding._keras_mask
    print(embedding.shape)
    # print(embedding[0])
    print(embedding_mask)

    print("\n=== GRU ===")
    GRU_seq    = tf.keras.layers.GRU(5, return_sequences=True, return_state=False)
    GRU_no_seq = tf.keras.layers.GRU(25, return_sequences=False, return_state=False)
    # LSTM = tf.keras.layers.LSTM(128)
    gru_seq = GRU_seq(embedding, mask=None, training=False, initial_state=None)
    gru_no_seq = GRU_no_seq(embedding, mask=None, training=False, initial_state=None)
    # lstm = LSTM(embedding, mask=None, training=False, initial_state=None)
    print('input shape:{}'.format(embedding))
    print('output gru_seq shape:{}'.format(gru_seq))
    print('output gru_no_seq shape:{}'.format(gru_no_seq))
    # print('output LSTM shape:{}'.format(lstm))


pass

if __name__ == '__main__':
    main()
pass 