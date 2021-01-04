Regularization of Tensorflow
==
Several information indicate that we should take care of the weights for regularization, such as the normalizing layers (batch normalization, layer normalization, etc.). *Because some of the weights should not be closed to zero.* For example, if the beta and gamma of batch normalization is 0, the operation would be wrong.

### ref 
* https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994

# the proposed solution in TF2
Since the TF2 use the Keras module, using the build-in operation would be a way. But build-in method make the [decoupling regularization](https://arxiv.org/abs/1711.05101) hard to be implmented. So, using the decoupleing weight decay or the build-in weight decay operation is also an issue. <p>
Since the weight decay can provide [effective learning rate](https://arxiv.org/abs/1706.05350) This example will use the build-in as example for weight decay purpose. <p>
(But there is still a [counter evidence](https://blog.janestreet.com/l2-regularization-and-batch-norm/) indicates the weight decay is helpless for training if normalization layer exists.)