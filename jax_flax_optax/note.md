# 置換model的weights
jax使用pytree進行管理，建議使用的工具是 jax.tree_util。幾個常用的工具:
1. jax.tree_util.tree_leaves ==> 取的節點的數值，可用在regularization上。如果直接置換kernel.value會造成梯度無法追蹤而報錯
2. nnx.Conv的replace方法，例如
```python
# 把kernel置換成0
a = flax.nnx.Conv(3,3,3)
a.kernel = a.kernel.replace(flax.nnx.Param(jnp.zeros(a.kernel.shape, a.kernel.dtype)))
```

# 操作模型中變數的原則
1. nnx.Param 跟 模型計算時產生的中介tensor ，兩者要區分清楚。尤其在做loss的時候 => 判斷原則就是"該計算究竟有沒有保存計算圖的必要"
2. 所有的nnx.Param處理，全部都要透過jax.tree_utils做處理(jax.tree_utils.map, jax.tree_utils.tree_leaves, jax.tree_utils.reduce)
3. 所有對nnx.Param的處理，都必須"即時"處理，不能存放中介(例如用self.xxx等等的變數保存中介結果)
4. 中介tensor要透過物件當中的變數(e.g self.xxxx)傳出，無法透過即時呼叫
