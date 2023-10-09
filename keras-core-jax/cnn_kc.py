import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np 
import jax 
import jax.numpy as jnp

# change the backend of keras to jax
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_core as kc  # noqa: E402

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

def main():
    bs = 32
    # opt = kc.optimizers.AdamW(global_clipnorm = 1.0)
    
    # learning rate shedule
    totalSteps = 5000
    decaySteps = 1000
    warmupSteps = 1000
    initialLearningRate = 0.0
    warmupTarget = 1e-4
    alpha = 2e-6
    learningRateSchedules = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = initialLearningRate,
            decay_steps = decaySteps,
            alpha = alpha,
            warmup_target = warmupTarget,
            warmup_steps = warmupSteps
            )
    
    # creating dataset iterator
    ds = tfds.load('mnist', split="train", shuffle_files=True)
    ds = ds.shuffle(1024).batch(32).repeat().prefetch(tf.data.AUTOTUNE)
    dsIter = iter(ds)
    
    # call the cnn model
    model = cnn()
    
    # jax 作為backend，在keras core會把GPU記憶體損耗光，目前原因未知。(v0.1.5, v0.1.6)
    # 解汙方法：把需要用到GPU記憶體的操作都先做完，最後再來宣告optimizer
    opt = kc.optimizers.AdamW(global_clipnorm = 1.0)
    
    # training loop
    # 1. 模型訓練時，會有trainable與non-trainable parameter，差別在於update時，是否會
    #    有梯 度流入。
    # 2. non-trainable並非不會被updated。最明顯的例子就是batch normalization有2個
    #    non-trainable parameters，分別是
    #   平均值和標準差。但是每次在計算的時候，會透過moving average進行updating，而不是
    #    使用梯度。
    # 3. stateless的fucntion，每次運算時需要把trainable和non-trainable都一起注入到
    #    optimizor。原因就是non-trainable variables有可能還是會在inferencing過程中被
    #    updated。 
    
    # define the cross-entropy as loss
    # by using the stateless calling
    def compute_loss(trainableVars, nonTrainableVars, img, lab):
        # 如果確認每一個non-trainable variable都不會更動，應該可以直接使用model計算。
        # 但模型中有non-trainable variable，而且該non-trainable variables會被updated，
        # 就需要用statless來回傳已經被updated的non-trainable variables。
        # 換句話說，使用stateless calling的寫法，泛用性比直接呼叫大很多。
        #
        # stateless function 在 Keras core 有三種。詳細狀況可參閱:
        # https://keras.io/keras_core/announcement/
        
        predOutput, updatedNonTrainableVars = model.stateless_call(trainableVars, nonTrainableVars, img)  # noqa: E501
        # pred = predOutput.primal
        # ce = lab * kc.ops.log(pred)
        # ce = kc.ops.sum(ce, axis=-1)
        # meanCE = kc.ops.mean(ce)
        # loss = -meanCE
        
        loss = kc.losses.CategoricalCrossentropy(from_logits=False)(lab, predOutput)
        
        return loss, updatedNonTrainableVars
    
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True) # 使用jax.value_and_grad的has_aux=True，才能取得non-trainable參數的結果  # noqa: E501
    
    # define the training step
    # @jax.jit
    def trainer(state, img, lab):
        sTrainableVars, sNonTrainableVars, sOptVars = state
        (loss, sNonTrainableVars), grads = grad_fn(sTrainableVars, sNonTrainableVars, img, lab)  # noqa: E501
        sTrainableVars, sOptVars = opt.stateless_apply(grads, sTrainableVars, sOptVars)
        return loss, (sTrainableVars, sNonTrainableVars, sOptVars)
    
    # training loop
    opt.build(model.trainable_variables)
    trainableVars = model.trainable_variables
    nonTrainableVars = model.non_trainable_variables
    optVars = opt.variables
    state = (trainableVars, nonTrainableVars, optVars)
    
    for step in range(totalSteps):
        # setting the training env variables
        dsFetcher = next(dsIter)
        cImg = (tf.cast(dsFetcher['image'], tf.float32).numpy() - 128) / 128
        cLab = jax.nn.one_hot(dsFetcher['label'].numpy(), 10)
        opt.lr = learningRateSchedules(step)
        
        # training 
        loss, state = trainer(state, cImg, cLab)
        exit()
        
        
        
    

if __name__ == "__main__":
    main()    
    
