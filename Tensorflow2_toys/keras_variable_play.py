import numpy as np
import tensorflow as tf 

### basic part ###
print('=== basic part start ===')
# make a simple model which has only 2 layers
Input = tf.keras.layers.Input([32, 32, 3])
out = tf.keras.layers.Conv2D(8, [3, 3])(Input)
print(out)

# announce the keras model using the specific input layers and output layers
model = tf.keras.Model(inputs=Input, outputs=out)
print(model)
# exit()

# inferencing using a dummy batch which contained only zeros
dummy_batch = np.zeros([10, 32, 32, 3])
# print(model(dummy_batch))
dummy_outputs = model(dummy_batch)
print(dummy_outputs.shape)

print('=== basic part finish ===')

### how to assign the weights into keras ###
print('=== assign part start ===')
# show the weights in the model
print(model.weights)
# we can also fetch the results into the numpy array
weight_fetcher = model.weights
print(weight_fetcher[1].numpy()) # index 1 is the "bias" which only has 8 elements
# we can also manipulate the weights using tensorflow, such as +1 
weight_fetcher[1].assign_add([1,1,1,1,1,1,1,1])
print(weight_fetcher[1].numpy())
# assign the variable by what you want is also possible
weight_fetcher[1].assign(np.random.random([8]))
print(weight_fetcher[1].numpy())
# list the trainable variables
print(model.trainable_variables)
print('=== assign part finish ===')


## The automatical function to extract the trainable variables in the tf keras model
print('== the keras variables playground start ==')
def cnn_trainable_variables(keras_model):
    model = keras_model
    total_variables_of_model = 0
    for i in model.trainable_variables:
        total_variables_of_model += sum(i.shape)
    pass
pass


print(cnn_trainable_variables(model))


print('== the keras variables playground finish ==')