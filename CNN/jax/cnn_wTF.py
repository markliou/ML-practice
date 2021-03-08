import tensorflow as tf 
import tensorflow_datasets as tfds 
import jax 
from jax.experimental import optimizers
import numpy as np

# jax 為了再現性，並同時遵守亂數可以產生的安全性，故用了hash。文件中並沒有詳細說明這個hash怎麼跑跟運作
# 但每次需要呼叫亂數的時候都須要給一個長度為2的陣列當作key，每次都更換key才能產生不同亂數。jax的說明文
# 建議使用jax.random.split來更換key。
global prngkey
prngkey = jax.random.PRNGKey(20) 

# 使用just in time (jit) decorator可以加速編譯
@jax.jit 
def conv_opt(x, _w):
    return jax.lax.conv_general_dilated(x, # 設定跟tensorflow一樣。如果原本是NCHW可以用 jax.lax.transpose(x, [0, 3, 1, 2])
                                        _w, # NHWC對應的HWIO
                                        [2, 2], # stride，對應為HW。應該是長度為n的tuple
                                        "SAME", # padding。有"SAME"跟"VALID"
                                        dimension_numbers=('NHWC','HWIO','NHWC') # 預設為None，對應的就是(‘NCHW’, ‘OIHW’, ‘NCHW’)。
                                        )
pass 

def conv2d_relu(x, weights_container, out_channel_no, kernel_size, layer_index = -1):
    global prngkey
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

    _w, _x, _b = None, None, None
    
    if layer_index == -1 : # if the layer is not be built, the weights should be given
        prngkey, c_prngkey = jax.random.split(prngkey) 
        # create the kernels
        _in_channel = x.shape[-1] # get the input channel number
        # rpn_key = jax.random.PRNGKey(np.random.randint(65536))
        rpn_key = np.random.randint(65536)

        _w = jax.nn.initializers.glorot_uniform()(c_prngkey, [kernel_size, kernel_size, _in_channel, out_channel_no]) # HWIO
        _x = conv_opt(x, _w)
        _b = jax.numpy.zeros(_x.shape) # bias

        # storing the weights
        weights_container.append([_w,_b])
    else:
        _w, _b = weights_container[layer_index]
        _x = conv_opt(x, _w)
    pass
    
    return jax.nn.relu(_x + _b)
pass

def loss(pred, truth): # * remember to give the log_softmax *
    return jax.numpy.mean(-1 * truth * pred) # negative log-likihood 
pass 

def one_hot(x):
    return jax.numpy.array(tf.one_hot(x, 10).numpy().astype('float32'))
pass

def main():

    dataset = tfds.load('mnist',  shuffle_files=True)
    tr, ts = iter(dataset['train'].batch(32).prefetch(1).repeat()), iter(dataset['test'].batch(1).repeat())
    # print("{} {}".format(tr.__next__()['image'].shape, tr.__next__()['label'].shape))
    tr_c = tr.__next__()

    ## build the container for keep the weights
    weights_container = []

    ## building the CNN
    feature_map_nos = [16, 32, 64, 10]
    # image = tr_c['image'].numpy()
    image = jax.numpy.ones([1, 28, 28, 1], dtype=np.float32) # give a dummy array for initialization
    output = (jax.numpy.array(image.astype('float32')) / 128.0) -1
    for feature_map_no in  feature_map_nos:
        output = conv2d_relu(output, weights_container, feature_map_no, 3, layer_index = -1)
    pass 
    print(output.shape)
    # using max-pooling 
    output = jax.numpy.reshape(output, [-1, output.shape[1] * output.shape[2], 10])
    output = jax.numpy.max(output, axis=-2)
    output = jax.nn.log_softmax(output)
    print(output.shape)

    ## training loop!!
    learning_rate = 1e-4

    # 1) jax.experimental.optimizer必須要匯入到主命名空間中，所以需要使用到import ... from
    # 2) optimizer物件在呼叫以後會返回三個不同的方法:
    #    2.1) opt_init : 用來初始化optimizer用的方法。這邊需要告訴optimizer要最佳化那些變數。
    #         因此會出現類似opt_init(param)的使用方式。param就是要用這個optimizer做最佳化的參數。
    #         opt_init會依照param建立在optimizer中相對應的參數，例如momentum或是目前的beta之類的。
    #         這時就須要把這些狀態存起來讓下次update的時候使用這些狀態參數。
    #    2.2) opt_update : 把梯度跟optimizer的狀態輸入後進行weights的update。optimizer的狀態就
    #         包括momentum這些之前記憶下來的數字。
    #    2.3) opt_param: 回傳會被最佳化的weights
    
    ## Create the optimizer, and get the essential objects
    opt_init, opt_update, get_param = optimizers.adam(learning_rate)
    
    ## initializing the optimizer, and the the status objects after initialing it
    opt_status = opt_init(weights_container)

    for training_step in range(5000):

        pass 

    print(loss(output, one_hot(tr_c['label'])))
    
pass 

if __name__ == "__main__":
    main()
pass



