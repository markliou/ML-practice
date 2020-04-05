Tensorflow Lite sample
===
This chapeter aim to practice the Tensorflow Lite.\
Since TF 2.2 is already GA, this chapter also use the **TF2** for the example. 

# training and model building
The training code is *cnn.py* which contain a simple classification task on MNIST. \
After training, this script will create a 'saved_model.pb' at the current folder.\
A tflite model and quantized tflite model is also built.

# inferencing code
the *cnn_tfl_infer.py* can load the qcnn.tflite which is create by cnn.py. \
After loading the tflite model, this script will check the input shape of this model and then creating a random tensor according to the checking. \
The random tensor will be feed into the model. The output will be printed on the screen.