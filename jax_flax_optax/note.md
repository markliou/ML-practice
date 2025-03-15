# 置換model的weights
jax使用pytree進行管理，建議使用的工具是 jax.tree_util。幾個常用的工具:
1. jax.tree_util.tree_leaves ==> 取的節點的數值，可用在regularization上。如果直接置換kernel.value會造成梯度無法追蹤而報錯
2. nnx.Conv的replace方法，例如
```python
# 把kernel置換成0
a = flax.nnx.Conv(3,3,3)
a.kernel = a.kernel.replace(flax.nnx.Param(jnp.zeros(a.kernel.shape, a.kernel.dtype)))
```
