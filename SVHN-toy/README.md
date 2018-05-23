# SVHN-toy
This toy script use the intutive thought to train a CNN which can reconize the SVHN dataset.

## small notes
1. This CNN is not easy to train. From some information, the CNN would need to be trained for 32 epoches ( or above).
2. In my case, I used the batch size of 32 and trained it for 50000 iterations (**about 150 epoches**) and the CNN start to get the right information.

```
example output:

['25544.png', '27590.png']
pred:[array([[0, 0, 0, 0, 1, 6],
       [0, 0, 0, 0, 0, 7]])]
ans :[array([[0, 0, 0, 0, 1, 6],
       [0, 0, 0, 0, 0, 7]])]
step:50670 loss:8.594249725341797

where pred means the output of the CNN and ans is the ground truth.
```
![25544.png](25544.png)
![27590.png](27590.png)

# license
MIT license