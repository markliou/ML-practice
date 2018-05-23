# SVHN-toy
This toy script use the intutive thought to train a CNN which can reconize the SVHN dataset.

## small notes
1. This CNN is not easy to train. From some information, the CNN would need to be trained for 32 epoches ( or above).
2. In my case, I used the batch size of 32 and trained it for 50000 iterations (**about 150 epoches**) and the CNN start to get the right information.

```
example output:

['18732.png', '28740.png']
pred:[array([[0, 0, 0, 0, 5, 2],
       [0, 0, 0, 2, 0, 1]])]
ans :[array([[0, 0, 0, 0, 5, 2],
       [0, 0, 0, 2, 0, 1]])]
step:50600 loss:6.863088607788086

```
[18732.png](18732.png)
[28740.png](28740.png)

# license
MIT license