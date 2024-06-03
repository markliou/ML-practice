tips
===

重新思考單純的policy gradient在reward上的表現與Q learning的差別。

* Q-learning 記錄了actions history與對應的transition state。如果單純的policy gradient並不會強制要求。最後結果就是在複雜行為中，會redundent到"最簡單的Action狀態"。以SpaceInvader的情況來說，就會一直按左開火或是右開火，讓飛機躲在角落一直開火。整個行為缺乏靈活性。

# 主要解決方式

1. 給予"state history": 這邊最簡單的作法就是把state的畫面狀態直接疊加，並且做移動平均。
2. 給予"action history": 這邊最簡單的作法就是把action當成一個雜訊產生器，然後對進行移動平均進行疊加。
