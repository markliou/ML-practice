import tensorflow as tf 
import numpy as np 

def AE_encoder():
    x = tf.keras.Input([28, 28, 1])
    e1 = tf.keras.layers.Conv2D(32, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(x)
    e2 = tf.keras.layers.Conv2D(32, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(e1)
    e3 = tf.keras.layers.Conv2D(32, (3,3), padding="SAME", strides=(2,2), activation=tf.nn.relu)(e2)
    out = tf.keras.layers.Conv2D(32, (3,3), padding="SAME", strides=(2,2), activation=None)(e3)
    return tf.keras.Model(Inputs=x, outputs=out)
pass 

def AE_decoder():
    pass 


def main():
    en = AE_decoder()
    print(en.summary)
pass 


if __name__ == "__main__":
    main()
pass 