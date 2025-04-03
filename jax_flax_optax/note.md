# 置換model的weights
jax使用pytree進行管理，建議使用的工具是 jax.tree_util。幾個常用的工具:
1. jax.tree_util.tree_leaves ==> 取的節點的數值，可用在regularization上。如果直接置換kernel.value會造成梯度無法追蹤而報錯
2. nnx.Conv的replace方法，例如
```python
# 把kernel置換成0
a = flax.nnx.Conv(3,3,3)
a.kernel = a.kernel.replace(flax.nnx.Param(jnp.zeros(a.kernel.shape, a.kernel.dtype)))
```

# 兩個模型互相置換 => 如果如一用leaves置換速度太慢
如果兩個模型形狀相同，直接透過state跟filter把變數取出後操作即可
思考邏輯: 
1. 解開state
2. 從state中取得parameter tree
3. 修改paramter tree
4. 把新的paramter tree跟原本的state進行merge => 必須要用merge的念建
5. 把新的state跟model進行update
```python
modelAState = flax.nnx.state(modelA)
modelBState = flax.nnx.state(modelB)
modelAParams = modelAState.filter(flax.nnx.Param)
modelBParams = modelBState.filter(flax.nnx.Param)
newTree = jax.tree_util.tree_map(lambda x, y: x * .9 + y * .1,
                             modelAParams, modelBParams)
newState = flax.nnx.State.merge(modelAState, newTree)
flax.nnx.update(modelA, newState)
```

# 操作模型中變數的原則 => 釐清"intermediate"觀念 self.sow(flax.nnx.Intermediate, "vqLoss", vqLoss, reduce_fn=lambda x, y: y)
1. nnx.Param 跟 模型計算時產生的中介tensor ，兩者要區分清楚。尤其在做loss的時候 => 判斷原則就是"該計算究竟有沒有保存計算圖的必要"
2. 所有的nnx.Param處理，全部都要透過jax.tree_utils做處理(jax.tree_utils.map, jax.tree_utils.tree_leaves, jax.tree_utils.reduce)
3. 所有對nnx.Param的處理，都必須"即時"處理，不能存放中介(例如用self.xxx等等的變數保存中介結果)
4. 中介tensor要透過物件當中的變數(e.g self.xxxx)傳出，無法透過即時呼叫
