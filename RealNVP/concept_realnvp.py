import tensorflow as tf 
import tensorflow_datasets as tfds 
import numpy as np 
import flow_latent_ops as flo


def affine_coupling_block():

    # log Jacobian determinant 在可逆的狀況下，會是"下三角形"，且在對角線之外的數值應該都是0。
    # 因此只需要把log(s)加起來就行了。 
    # ref1. https://proceedings.neurips.cc/paper/2016/hash/ddeebdeefdb7e7e7a697e1c3e3d8ef54-Abstract.html
    # ref2. https://srbittner.github.io/2019/06/26/normalizing_flows/

    pass  

def RealNVP(): 
    x = tf.keras.Input(shape=[7, 7, 16])
    out = 0
    return tf.keras.Model(inputs=x, outputs=out)
pass 

def batchSqeeze2DFeatureMap(x):
    x_x = x
    x_x /= 128 
    x_x -= 1
    return flo.sqeeze2DFeatureMap(x_x, steps=4)
pass

def main(): 

    ## basic parameter zone 
    batch_size = 30
    training_interation_no = 5000 
    learning_rate = 1E-4 
    optimizer = tf.keras.Optimizer.RMSPropOptimizer(learning_rate, clipnorm=1)


    # split dataset into training and test sets, creat the generators
    (kmnist_tr, kmnist_ts) = tfds.load('kmnist', 
                                        split=['train','test'],
                                        shuffle_files=True,
                                        as_supervised=False,
                                        with_info=False,)

    kmnist_tr = kmnist_tr.batch(30).repeat()
    kmnist_tr_iter = iter(kmnist_tr)

    # build the RealNVP
    model = RealNVP()

    for training_iter_no in range(training_iter_no):
        sq_tr = tf.map_fn(fn=batchSqeeze2DFeatureMap, elems=tr)
        print(sq_tr)
    pass 
pass  

if __name__ == "__main__":
    main() 
pass 