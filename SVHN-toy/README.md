# SVHN-toy
This toy script use the intutive thought to train a CNN which can reconize the SVHN dataset.

## small notes
1. This CNN is not easy to train. From some information, the CNN would need to be trained for 32 epoches ( or above).
2. In my case, I used the batch size of 32 and trained it for 50000 iterations (**about 150 epoches**) and the CNN start to get the right information.

```
example output:
pred:[array([[0, 0, 0, 0, 3, 4],
       [0, 0, 0, 0, 1, 9]])]
ans :[array([[0, 0, 0, 0, 3, 4],
       [0, 0, 0, 2, 1, 9]])]
step:50575 loss:7.756442070007324

```

# license
MIT license