Tensorflow 2 model to Tensorflow 1 mode
==
Currently, many NN compilers, such as TVM or Openvino. This means if we use TF2 training model, this model could be not transformed into the high performance codes while running on NN chips. <p>
To solve this, transforming the TF2 model into TF1 format would be nessessary. <p>
Since Keras become more important in TF2, using Keras API to solving this problem would be feasible. <p>
This repo try to convert the model made from TF2 to transform into TF1 format using tf.keras API. <p>

# model files
The model is made using TF2:
* k_model.h5
* reference_codes/*.py : this folder contain the training and inferencing codes which can provide the model information.

# TF2_model_loader.py
This file will load the models trained using TF2.2 and try to save it into TF 1.14 using .pb format.


# references
https://www.codeleading.com/article/57322536101/