import tensorflow as tf 
import numpy as np 
import jax 
import jax.numpy as jnp

# change the backend of keras to jax
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_core as kc


def cnn():
    x = kc.layers.Input([28, 28, 1])
    c1 = kc.layers.Conv2D(32, (3,3), padding="same", activation=kc.activations.elu)(x)
    c2 = kc.layers.Conv2D(32, (3,3), padding="same", activation=kc.activations.elu)(c1)
    
    flatten = kc.layers.Flatten()(c2)
    
    fc1 = kc.layers.Dense(512, kc.activations.mish)(flatten)
    fc2 = kc.layers.Dense(256, kc.activations.mish)(fc1)
    out = kc.layers.Dense(10, kc.activations.mish)(fc2)
    
    return kc.Model(inputs=x, outputs=out)
    

def main():
    model = cnn()
    print(model.summary())

if __name__ == "__main__":
    main()    
    
