import tensorflow as tf 
import tensorflow_datasets as tfds 
import jax 

dataset = tfds.load('mnist',  shuffle_files=True)
tr, ts = iter(dataset['train'].batch(32).prefetch(1).repeat()), iter(dataset['test'].batch(1).repeat())
print("{} {}".format(tr.__next__()['image'].shape, tr.__next__()['label'].shape))