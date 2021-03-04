import tensorflow as tf 
import tensorflow_datasets as tfds 
import jax 
import numpy as np

def conv2d_relu(x, out_channel_no, kernel_size, weights_container, layer_index = None):
    # background:
    # jax中內建convolution的operator共有目前有兩個:
    # stax : 有比較齊全的神經網路操作，包括dropout等等。但目前還在experimental的模組下 (https://jax.readthedocs.io/en/latest/jax.experimental.stax.html)
    # lax : 只有一些比較基本的模組。但官網上的教學使用這個模組。所以先用這模組玩看看。 (https://jax.readthedocs.io/en/latest/jax.lax.html)
    # 
    # operator comparing:
    # lax.conv是convolution的基本型態，但有把功能寫更全的lax.conv_general_dilated。
    # 建議使用 lax.conv_general_dilated ，除了文件比較齊全之外，功能也比較齊全。
    # 例如可以對input做dialation來達成transpose convolution的功能
    #
    # notes for lax.conv_general_dilated
    # 1) tensor操作的行為: default shape=>(input, kernel, output)=(‘NCHW’, ‘OIHW’, ‘NCHW’)，
    #    相較於tf常用的shape => (input, kernel, output)=(‘NHWC’, ‘HWIO’, ‘NHWC’)。
    #    因此可以先用lax.transpose先做轉置。並且在 dimension_numbers 設定 (‘NHWC’, ‘HWIO’, ‘NHWC’)
    # 2) 透過修改lhs_dialation跟rhs_dialation達成transpose convolution及atrous convolution
    # 3) feature_group_count及batch_group_count主要用在xla上
    # 4) 說明文件中的 n 定義為會進行"stride"的維度。例如圖像是2D陣列，所以進行convolution的方向會是XY兩個方向，所以
    #    n就等於2。1D convolution只有x方向所以n=1。這種做法讓convolution可以往多維度移動，而input的shape是n+2表示
    #    會多出兩個維度。這兩個維度就是[batch_size, output_channel]

    # create the kernels
    _in_channel = x.shape[-1] # get the input channel number
    # rpn_key = jax.random.PRNGKey(np.random.randint(65536))
    rpn_key = np.random.randint(65536)

    _w = jax.nn.initializers.glorot_uniform()(rpn_key, [kernel_size, kernel_size, _in_channel, out_channel_no]) # HWIO
    
    _x = jax.lax.conv_general_dilated(x, # 設定跟tensorflow一樣。如果原本是NCHW可以用 jax.lax.transpose(x, [0, 3, 1, 2])
                                      _w, # NHWC對應的HWIO
                                      [1, 1], # stride，對應為HW。應該是長度為n的tuple
                                      "SAME", # padding。有"SAME"跟"VALID"
                                      dimension_numbers=('NHWC','HWIO','NHWC') # 預設為None，對應的就是(‘NCHW’, ‘OIHW’, ‘NCHW’)。
                                     )

    _b = jax.numpy.zeros(_x.shape) # bias
    
    # storing the weights
    weights_container.append(_w)
    weights_container.append(_b)

    return jax.nn.relu(_x + _b)
pass

def main():

    # dataset = tfds.load('mnist',  shuffle_files=True)
    # tr, ts = iter(dataset['train'].batch(32).prefetch(1).repeat()), iter(dataset['test'].batch(1).repeat())
    # print("{} {}".format(tr.__next__()['image'].shape, tr.__next__()['label'].shape))

    sample = jax.numpy.ones([10, 28, 28, 1])
    
    # build the container for keep the weights
    weights_container = []

    print(conv2d_relu(sample, 64, 3, weights_container))

    print(weights_container)
pass 

if __name__ == "__main__":
    main()




