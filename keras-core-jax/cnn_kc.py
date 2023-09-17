import tensorflow as tf 
import tensorflow_datasets as tfds
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
    bs = 32
    opt = kc.optimizers.AdamW(global_clipnorm = 1.0)
    
    # learning rate shedule
    total_steps = 5000
    decay_steps = 1000
    warmup_steps = 1000
    initial_learning_rate = 0.0
    warmup_target = 1e-4
    alpha = 2e-6
    lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = initial_learning_rate,
            decay_steps = decay_steps,
            alpha = alpha,
            warmup_target = warmup_target,
            warmup_steps = warmup_steps,
            )
    
    # creating dataset iterator
    ds = tfds.load('mnist', split="train", shuffle_files=True)
    ds = ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
    
    model = cnn()
    
    def loss():
        pass
    

if __name__ == "__main__":
    main()    
    
