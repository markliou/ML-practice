TVM on Tensorflow
==
This practice try to make the tensorflow model to TVM model

# Enviroment -- Docker
The docker image can be get:
```
docker pull markliou/tvm
```
This image is created using nvidia/cuda as the base. The TVM is installed using source code and CUDA support. Hence, remember to run this image with nvidia-docker

# toy neural network
The example is from the "TensorflowLite" section. The model use MNIST as input, 10 class as output. The detail is listed below:
```python
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
def conv_net(x):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    cnn1 = tf.keras.layers.Conv2D(64, 3, strides=(1,1), activation=tf.nn.relu)(x)
    cnn2 = tf.keras.layers.Conv2D(64, 3, strides=(2,2), activation=tf.nn.relu)(cnn1)
    cnn3 = tf.keras.layers.Conv2D(128, 3, strides=(1,1), activation=tf.nn.relu)(cnn2)
    cnn4 = tf.keras.layers.Conv2D(128, 3, strides=(2,2), activation=tf.nn.relu)(cnn3)
    fc0 = tf.keras.layers.Flatten()(cnn4)
    fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(fc0)
    fc2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(fc1)
    out = tf.keras.layers.Dense(10)(fc2)
    return out
pass
#Construct model
logits = conv_net(X)
prediction = tf.nn.softmax(logits)
```
This script will train a model and freeze it into .pb file.

# compile the .pb into TVM IR
The detail is recorder in *tvm_converter.py* which will generate:
1. .json
2. .params 

# inference using C compiler
需要額外安裝三個必要的套件
1. [dlpack](https://github.com/dmlc/dlpack) 
2. [dmlc](https://github.com/dmlc/dmlc-core) (但dmlc其實在tvm底下的3rdparty資料夾下面有，也可以直接使用)
套件變動很大，底下雖然有其它人寫的教學，但路徑跟套見的使用在後來的版本有很多變動。如果想看範例，可以從tvm/apps/how_to_deploy資料夾下來看。重點是Makefile檔的修改。最主要需要修改的內容如下:
1. TVM的安裝路徑。因為範例是在apps資料夾下，所以只需要網上跑兩層就可以取得tvm。這邊就要看使用時是甚麼狀況。
2. 重點是給予tvm runtime，也就是 *tvm_runtime_pack.cc* 這個程式。進入後也記得修改裡面include library的路徑。
3. 要把一開始在python裡面編出來的.so檔(這邊是cnn_tvm_lib.so)在編譯階段也要放進去。

以下是Makefile經過修改後的節果
```=
# Makefile Example to deploy TVM modules.
TVM_ROOT="/root/tvm"
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++14 -O2 -fPIC\
        -I${TVM_ROOT}/include\
        -I${DMLC_CORE}/include\
        -I${TVM_ROOT}/3rdparty/dlpack/include\

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread

.PHONY: clean all

all: tvm_c_inference

# Build rule for all in one TVM package library
libtvm_runtime_pack.o: tvm_runtime_pack.cc
        $(CXX) -c $(PKG_CFLAGS) -o $@  $^

# Deploy using the all in one TVM package library
tvm_c_inference: tvm_c_inference.c libtvm_runtime_pack.o cnn_tvm_lib.so
        $(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

clean:
        rm -fr libtvm_runtime_pack.o tvm_c_inference
```
請注意在tvm_c_inferece那段描述的地方，最後一個是有放入cnn_tvm_lib.so。沒有放會出現unlink的錯誤。

# reference
https://github.com/starmee/AI-Notes/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E9%83%A8%E7%BD%B2%E6%A1%86%E6%9E%B6/TVM/TVM%E9%83%A8%E7%BD%B2.md#build-tensorflow