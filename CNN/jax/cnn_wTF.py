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
    
    return jax.lax.conv_general_dilated(x, # 設定跟tensorflow一樣。如果原本是NCHW可以用 jax.lax.transpose(x, [0, 3, 1, 2])
                                        _w, # NHWC對應的HWIO
                                        [2, 2], # stride，對應為HW。應該是長度為n的tuple
                                        "SAME", # padding。有"SAME"跟"VALID"
                                        dimension_numbers=('NHWC','HWIO','NHWC') # 預設為None，對應的就是(‘NCHW’, ‘OIHW’, ‘NCHW’)。
                                        )
pass 

def jax_fcn_init(weights_container, input_tensor_shape, out_channel_nos, kernel_size = 3):
    global prngkey
    
    # _w, _x, _b = None, None, None
    _x = jax.numpy.ones(input_tensor_shape, dtype=np.float32) # give a dummy array for initialization
    
    for out_channel_no in out_channel_nos:
        prngkey, c_prngkey = jax.random.split(prngkey) 
        # create the kernels
        _in_channel = _x.shape[-1] # get the input channel number
        
        _w = jax.nn.initializers.glorot_uniform()(c_prngkey, (kernel_size, kernel_size, _in_channel, out_channel_no)) # HWIO
        _x = conv_opt(_x, _w)
        _b = jax.numpy.zeros(_x.shape) # bias
        
        # storing the weights
        weights_container.append((np.array(_w),np.array(_b))) 
    pass
pass

@jax.jit
def loss(weights_container, x, truth): # * remember to give the log_softmax *
    # 寫 loss 的需要特別注意，jax在計算gradient的時候會以"輸入第一個參數"作為標的計算梯度。
    # 也就是如果把 x 放在第一個，這邊就會計算針對 x 的gradient。這邊的task是要針對 weigts，
    # 所以一定要把 wights_container 放成第一個輸入的參數才會得到正確答案。
    pred = jax_fcn(x, weights_container)
    return jax.numpy.mean(-1 * truth * pred) # negative log-likihood 
pass 

def accuracy(x, weights_container, truth):
    pred = np.argmax(jax_fcn(x, weights_container), axis=-1)
    return(np.mean(np.equal(pred.astype(dtype=np.int8),truth.astype(dtype=np.int8))))
pass 

def one_hot(y):
    return jax.numpy.array(tf.one_hot(y, 10).numpy().astype('float32'))
pass

@jax.jit
def jax_fcn(x, weights_container):
    ## forward computing
    _x = (jax.numpy.array(x.astype('float32')) / 128.0) -1
    for weights in weights_container[:-1]:
        _w, _b = weights
        _x = conv_opt(_x, _w)
        _x = jax.nn.relu(_x + _b)
    pass
    _w, _b = weights_container[-1]
    _x = conv_opt(_x, _w)
    # using max-pooling
    _x = jax.numpy.reshape(_x, [-1, _x.shape[1] * _x.shape[2], 10])
    _x = jax.numpy.max(_x, axis=-2)
    output = jax.nn.log_softmax(_x)
    #print(output.shape)
    
    return output
pass 

def main():

    dataset = tfds.load('mnist',  shuffle_files=True)
    tr, ts = iter(dataset['train'].batch(32).prefetch(1).repeat()), iter(dataset['test'].batch(1).repeat())
    # print("{} {}".format(tr.__next__()['image'].shape, tr.__next__()['label'].shape))

    ## build the container for keep the weights
    weights_container = []
    
    ## building the CNN
    # 建立網路的時候，在subroutine盡量不要放if。if有可能造成計算圖的斷裂。
    # 因此weight initialization跟forward兩個function盡量分開。這樣也同時可以使用到jit加速。
    feature_map_nos = [16, 32, 64, 10]
    input_shape = [32, 28, 28, 1]
    jax_fcn_init(weights_container, input_shape,feature_map_nos, 3)
    print('weights initialized ...')

    # test the forwarding
    # print(jax_fcn(jax.numpy.ones(input_shape, dtype=np.float32), weights_container))
    # exit()
    
    
    ##### creating the optimizer #####
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
    learning_rate = 1e-4
    opt_init, opt_update, opt_get_params = optimizers.adam(learning_rate)
    
    ## initializing the optimizer, and the the status objects after initialing it
    opt_status = opt_init(weights_container)
    # print(opt_get_params(opt_status))
    # exit()

    ##### training loop #####
    for training_step in range(5000):
        tr_c = next(tr)
        image, label = tr_c['image'].numpy(), tr_c['label'].numpy()
        image, label = image.astype('float32'), label.astype('float32')
        
        ## computing the loss and gradients
        current_loss, gradients = jax.value_and_grad(loss)(weights_container, image, one_hot(label))
        opt_status = opt_update(training_step, gradients, opt_status)
        weights_container = opt_get_params(opt_status)
        
        # print(gradients.shape)
        # print(opt_get_params(opt_status))
        # exit()

        print('step {} loss:{} accuracy:{}'.format(training_step, current_loss, accuracy(image, weights_container, label)))
    pass 

    
    
pass 

if __name__ == "__main__":
    main()
pass



