MAML
==
Using TF2 for training the MAML model.
The essential functions are listed:
1. get the weights
2. assign the weights

This example use the KMNIST for practice. You can also modify the example to MNIST easily.

# The task describe
The tasks are set as 2-classes classification. The neural network only need to tell which image are belonged to specified class. For example, if we have two images of 8 and 5, we define the 8 and 5 as class 0 and class 1. After updating the weights of the neural network, we expect the neural network can specify the 8 as class 0 and 5 as class 1 in the future tasks. 

# The parameters setting
* support set and query set: in this example, the same support set and query set are used. this would not be good in the practical task, but would be suitale for showing the example due to avoiding complex code
* ways and shots: this example use 2ways-1shot for training.

# refereces
可參考的程式碼: https://colab.research.google.com/github/mari-linhares/tensorflow-maml/blob/master/maml.ipynb#scrollTo=xzVi0_YfB2aZ