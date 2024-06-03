tips
===

重新思考單純的policy gradient在reward上的表現與Q learning的差別。

* Q-learning 記錄了actions history與對應的transition state。如果單純的policy gradient並不會強制要求。最後結果就是在複雜行為中，會redundent到"最簡單的Action狀態"。以SpaceInvader的情況來說，就會一直按左開火或是右開火，讓飛機躲在角落一直開火。整個行為缺乏靈活性。

### 主要解決方式

1. 給予"state history": 這邊最簡單的作法就是把state的畫面狀態直接疊加，並且做移動平均。
2. 給予"action history": 這邊最簡單的作法就是把action當成一個雜訊產生器，然後對進行移動平均進行疊加。

```python
preObservation = observation # for keeping states for the transition information
observation, reward, terminated, truncated, info = self.env[eipNo].step(action)
# 把動作歷史放到圖像裡面讓神經網路處理
observation = (np.array(observation) - 128.0)/128.0 * .6 + preObservation * .2 + np.random.normal(scale=(action / 6),  size=[210, 160, 3])  * .2
```

### 次要解決方式

* optimizer restart: 隔一段時間就把optimizer重置。從論文研究當中發現2階momentum對於已經改變的神經網路跟狀態有害。所以最好隔一段時間就重置。
* 大Batch size
* action entropy regularization: 增加熵的評估，讓模型在做決策時不會太"自信"
* 戰機位置切出來，獨立處理(在神經網路裡面)
* 防空位置切出來，獨立處理(在神經網路裡面)

### 不太有用的方式
* 強制給定不同位置有不同分數，意圖讓戰機停在畫面中央
* 強制比對批次中的Action，意圖讓action的變化性夠多。
