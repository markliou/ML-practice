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

# reference
https://github.com/starmee/AI-Notes/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E9%83%A8%E7%BD%B2%E6%A1%86%E6%9E%B6/TVM/TVM%E9%83%A8%E7%BD%B2.md#build-tensorflow