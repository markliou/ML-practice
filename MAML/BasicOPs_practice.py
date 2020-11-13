import tensorflow as tf 
import numpy as np

def cnn():
    Input = tf.keras.Input([28, 28, 1])
    conv1 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(Input) #[14,14]
    conv2 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(conv1) #[7,7]
    conv3 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", activation=tf.nn.relu)(conv2) #[4,4]
    fc = tf.keras.layers.Flatten()(conv3)
    fc1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc)
    fc2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc1)
    fc3 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(fc2)
    out = tf.keras.layers.Dense(1, activation=None)(fc3)

    return tf.keras.Model(inputs=Input, outputs=out)
pass 

def main():
    cnn_model = cnn()
    # print(cnn_model.summary())
    
    # _output = cnn_model.predict(np.random.randn(32,28,28,1).astype('float32'))
    # print(_output)

    _output = cnn_model(np.random.randn(32,28,28,1).astype('float32'))
    print(_output)

    print(cnn_model.weights) # this give the all weights in the model
    # cnn_model.save('model')

    target_weights = cnn_model.weights[0] # assign the weights needed to be handled
    ## get the weights and store them
    target_weights_keeper = target_weights.numpy()
    print(target_weights_keeper)

    ## assign the 1 as the weights to model
    target_weights.assign(np.ones(target_weights.shape.as_list()))
    print(target_weights)

    ## assign the previous weihts to the mode
    target_weights.assign(target_weights_keeper)
    print(cnn_model.weights)

pass 


if __name__ == "__main__":
    main()
pass 
