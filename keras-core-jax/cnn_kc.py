import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np 
import jax 
import jax.numpy as jnp

# change the backend of keras to jax
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_core as kc

# limited the gpu memory usage when TF working
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def cnn():
    x = kc.layers.Input([28, 28, 1])
    c1 = kc.layers.Conv2D(32, (3,3), padding="same", activation=kc.activations.elu)(x)
    c2 = kc.layers.Conv2D(32, (3,3), padding="same", activation=kc.activations.elu)(c1)
    
    flatten = kc.layers.Flatten()(c2)
    
    fc1 = kc.layers.Dense(512, kc.activations.mish)(flatten)
    fc2 = kc.layers.Dense(256, kc.activations.mish)(fc1)
    out = kc.layers.Dense(10, kc.activations.mish)(fc2)
    out = kc.layers.Softmax()(out)
    
    return kc.Model(inputs=x, outputs=out)
    
# loss function: softmax cross entropy
def loss(cImg, cLab, model):
    pred = model(cImg)
    ce = cLab * kc.ops.log(pred)
    ce = kc.ops.sum(ce, axis=-1)
    meanCE = kc.ops.mean(ce)
    return -meanCE

def main():
    bs = 32
    # opt = kc.optimizers.AdamW(global_clipnorm = 1.0)
    
    # learning rate shedule
    total_steps = 5000
    decay_steps = 1000
    warmup_steps = 1000
    initial_learning_rate = 0.0
    warmup_target = 1e-4
    alpha = 2e-6
    lRFn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = initial_learning_rate,
            decay_steps = decay_steps,
            alpha = alpha,
            warmup_target = warmup_target,
            warmup_steps = warmup_steps
            )
    
    # creating dataset iterator
    ds = tfds.load('mnist', split="train", shuffle_files=True)
    ds = ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
    dsIter = iter(ds)
    
    # call the cnn model
    model = cnn()
    
    # jax 作為backend，在keras core會把GPU記憶體損耗光，目前原因未知。(v0.1.5, v0.1.6)
    # 解汙方法：把需要用到GPU記憶體的操作都先做完，最後再來宣告optimizer
    opt = kc.optimizers.AdamW(global_clipnorm = 1.0)
    
    # training loop
    # 1. 模型訓練時，會有trainable與non-trainable parameter，差別在於update時，是否會有梯度流入。
    # 2. non-trainable並非不會被updated。最明顯的例子就是batch normalization有2個non-trainable parameters，分別是
    #   平均值和標準差。但是每次在計算的時候，會透過moving average進行updating，而不是使用梯度。
    # 3. stateless的fucntion，每次運算時需要把trainable和non-trainable都一起注入到optimizor。原因就是non-trainable
    #    variables有可能還是會在inferencing過程中被updated。 
    
    for step in range(total_steps):
        # setting the training env variables
        opt.lr = lRFn(step)
        dsFetcher = next(dsIter)
        cImg = (tf.cast(dsFetcher['image'], tf.float32) - 128) / 128
        cLab = jax.nn.one_hot(dsFetcher['label'].numpy(), 10)
        
        # jax需要先透過grad函數包裝loss函數，再透過第二段傳遞變數
        grad_fn = jax.value_and_grad(loss) # 使用jax.grad只會給出梯度。使用value_and_grad會同時給出梯度跟loss。
        grads = grad_fn(cImg, cLab, model)
        
        # apply the gradients
        opt.apply_gradients(grads)
        
        
        exit()
    

if __name__ == "__main__":
    main()    
    
