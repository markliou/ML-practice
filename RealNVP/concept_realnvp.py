import tensorflow as tf 
import tensorflow_datasets as tfds 
import tensorflow_probability as tfp
import numpy as np 
import flow_latent_ops as flo


def affine_coupling_block(x, blocks_logs, blocks_t, blockno, forward = True):
    layer_no = 8
    if forward:
        blocks_logs.append([])
        blocks_t.append([])
    pass
    
    def base_conv2d(blocks, blockno, layerno, output_channel_no = 64):
        if forward:
            if layerno == 0: # the new block need a list for containing the layers
                blocks[blockno] = []
            pass 

            conv = tf.keras.layers.Conv2D(output_channel_no, 
                                          [3, 3], 
                                          strides=(1, 1), 
                                          padding="SAME", 
                                          activation=tf.nn.relu)
            blocks[blockno].append(conv)
        else: # backward
            conv = blocks[blockno][layerno]
        pass
        return conv
    pass

    latent_logs = base_conv2d(blocks_logs, blockno, 0)(x)
    for layer_ind in range(1, layer_no):
        latent_logs = base_conv2d(blocks_logs, blockno, layer_ind)(latent_logs)
    pass
    latent_logs = tf.nn.tanh(base_conv2d(blocks_logs, blockno,  layer_no, 8)(latent_logs))

    latent_t = base_conv2d(blocks_t, blockno, 0)(x)
    for layer_ind in range(1, layer_no):
        latent_t = base_conv2d(blocks_t, blockno, layer_ind)(latent_t)
    pass
    latent_t = base_conv2d(blocks_t, blockno, layer_no, 8)(latent_t)

    # log Jacobian determinant 在可逆的狀況下，會是"下三角形"，且在對角線之外的數值應該都是0。
    # 因此只需要把log(s)加起來就行了。 
    # ref1. https://proceedings.neurips.cc/paper/2016/hash/ddeebdeefdb7e7e7a697e1c3e3d8ef54-Abstract.html
    # ref2. https://srbittner.github.io/2019/06/26/normalizing_flows/
    log_det = tf.reduce_sum(latent_logs, [-1])

    return [latent_logs, latent_t, log_det]
pass  

def RealNVP_forward(flow_block_no = 32): 
    x = tf.keras.Input(shape=[7, 7, 16])
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

    # (x, blocks_logs, blocks_t, blockno, forward = True)
    blocks_logs_1, blocks_t_1, blocks_logs_2, blocks_t_2 = [], [], [], []
    logdet_loss, el_loss = 0, 0
    distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(x1, dtype=tf.float32), scale_diag=tf.ones_like(x1, dtype=tf.float32))
    
    logs1, t1, logdet1 = affine_coupling_block(x2, blocks_logs_1, blocks_t_1, 0)
    logs2, t2, logdet2 = affine_coupling_block(x1, blocks_logs_2, blocks_t_2, 0)
    latent1 = x1 * tf.exp(logs1) + t1
    latent2 = x2 * tf.exp(logs2) + t2
    el_loss += (distribution.log_prob(latent1) + distribution.log_prob(latent2))
    logdet_loss += (logdet1 + logdet2)
    for flow_block in range(1, flow_block_no) :
        logs1, t1, logdet1 = affine_coupling_block(latent2, blocks_logs_1, blocks_t_1, flow_block)
        logs2, t2, logdet2 = affine_coupling_block(latent1, blocks_logs_2, blocks_t_2, flow_block)
        latent1 = latent1 * tf.exp(logs1) + t1
        latent2 = latent2 * tf.exp(logs2) + t2
        logdet_loss += (logdet1 + logdet2)
        el_loss += (distribution.log_prob(latent1) + distribution.log_prob(latent2))
    pass

    out = [tf.concat([latent1, latent2], axis=-1), -tf.reduce_mean(logdet_loss + el_loss)]
    return [tf.keras.Model(inputs=x, outputs=out), [blocks_logs_1, blocks_t_1, blocks_logs_2, blocks_t_2]]
pass 

def RealNVP_backward(flow_blocks): 
    blocks_logs_1, blocks_t_1, blocks_logs_2, blocks_t_2 = flow_blocks
    block_no = len(blocks_logs_1)
    x = tf.keras.Input(shape=[7, 7, 16])
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

    logs1, t1, logdet1 = affine_coupling_block(x2, blocks_logs_1, blocks_t_1, -1, forward=False)
    logs2, t2, logdet2 = affine_coupling_block(x1, blocks_logs_2, blocks_t_2, -1, forward=False)
    latent1 = x1 * tf.exp(logs1) + t1
    latent2 = x2 * tf.exp(logs2) + t2
    for flow_block in [b for b in range(1, block_no)[::-1]]:
        logs1, t1, logdet1 = affine_coupling_block(latent2, blocks_logs_1, blocks_t_1, flow_block, forward=False)
        logs2, t2, logdet2 = affine_coupling_block(latent1, blocks_logs_2, blocks_t_2, flow_block, forward=False)
        latent1 = (latent1 - t1) * tf.exp(logs1 * -1)
        latent2 = (latent2 - t2) * tf.exp(logs2 * -1)
    pass

    out = tf.concat([latent1, latent2], axis=-1)
    return tf.keras.Model(inputs=x, outputs=out)
pass

def batchSqeeze2DFeatureMap(x):
    # normalization
    x_x = tf.cast(x, dtype=tf.float32)
    x_x /= 128. 
    x_x -= 1.
    return flo.sqeeze2DFeatureMap(x_x, steps=4)
pass

def batchUnsqeeze2DFeatureMap(x):
    return flo.unsqeeze2DFeatureMap(x, steps=4, channel=1)
pass

def main(): 

    ## basic parameter zone 
    batch_size = 32
    training_iter_steps = 5000 
    learning_rate = 1E-4 
    optimizer = tf.keras.optimizers.RMSprop(learning_rate, clipnorm=1)


    # split dataset into training and test sets, creat the generators
    (kmnist_tr, kmnist_ts) = tfds.load('kmnist', 
                                        split=['train','test'],
                                        shuffle_files=True,
                                        as_supervised=False,
                                        with_info=False,)

    kmnist_tr = kmnist_tr.batch(batch_size).repeat().prefetch(4)
    kmnist_tr_iter = iter(kmnist_tr)

    # build the RealNVP
    model_forward, flow_blocks = RealNVP_forward()
    model_backward = RealNVP_backward(flow_blocks)
    # print(model.summary())

    for training_iter_step in range(training_iter_steps):
        tr = next(kmnist_tr_iter)
        sq_tr = tf.map_fn(fn=batchSqeeze2DFeatureMap, elems=tr['image'], dtype=tf.float32)

        def loss():
            # print(model(sq_tr))
            return model_forward(sq_tr)[1]
        pass 

        optimizer.minimize(loss, model_forward.trainable_variables)

        if training_iter_step % 1 == 0:
            print("step:{}  loss:{}".format(training_iter_step, loss()))
            tr_forward = model_forward(sq_tr)[0]
            tr_backward = model_backward(tr_forward)
            tr_forward = tf.map_fn(fn=batchUnsqeeze2DFeatureMap, elems=tr_forward, dtype=tf.float32)
            tr_backward = tf.map_fn(fn=batchUnsqeeze2DFeatureMap, elems=tr_backward, dtype=tf.float32)
            tf.keras.preprocessing.image.save_img("tr_forward.png", tr_forward[0],"channels_last")
            tf.keras.preprocessing.image.save_img("tr_backward.png", tr_backward[0],"channels_last")
            tf.keras.preprocessing.image.save_img("tr.png", tr['image'][0],"channels_last")
        pass
    pass 
pass  

if __name__ == "__main__":
    main() 
pass 