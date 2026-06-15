# JAX Atari Space Invaders 練習環境設定

本資料夾包含了使用 JAX 進行 Atari Space Invaders 強化學習（如 PPO）練習的環境設定與基礎程式碼。

遵循「測試歸測試、功能歸功能」的軟體工程原則，本目錄結構進行了精簡與分離。

---

## 1. Docker 環境與 Atari 安裝

您的 Docker 映像檔（Image）命名為：`markliou/jax-1080ti`。

為了支援 Atari 遊戲環境，[dockerfile](file:///home/markliou/ML-practice/ReinforcementLearning/Atari/jax-1080ti/devEnv/dockerfile) 已新增了 `gymnasium` 與 `ale-py` (包含授權同意，自動下載 ROM) 的安裝指令：

```dockerfile
# install gymnasium and atari packages, and automatically download/authorize Atari ROMs
RUN uv pip install --system "gymnasium[atari,accept-rom-license]" --break-system-packages
```

### 為什麼需要 `accept-rom-license`？
Atari 遊戲 ROM 受版權保護，RL 套件不能直接內建。使用 `accept-rom-license` 可讓安裝腳本在非互動式的 Docker 建置過程中自動下載並配置好遊戲環境，避免編譯中斷。

---

## 2. 檔案結構與說明

本專案將主體功能與測試邏輯完全分開：

### 專案功能模組：[space_invaders.py](file:///home/markliou/ML-practice/ReinforcementLearning/Atari/jax-1080ti/space_invaders.py)
- **用途**：提供供訓練腳本導入的 `JaxAtariEnv` 封裝類別。
- **特點**：
  - **純淨無雜質**：只包含環境初始化與 `reset` / `step` 轉換，不含任何列印或測試代碼，適合直接作為 Module 導入。
  - **JAX 原生整合**：`reset()` 與 `step()` 回傳的 Observation、Reward、Done 皆會自動轉為 `jax.numpy.array`，避免在您的演算法中手動轉換。
  - **Docker/Headless 友善**：預設使用 `rgb_array` 渲染模式，不會開啟任何視窗，完全相容於容器化環境。
  - **現代 Gymnasium 相容**：自動處理 `import ale_py` 以進行環境註冊，並將新版 Gymnasium 的 `terminated`/`truncated` 合併為標準的單一 `done` 訊號。

### 環境測試腳本：[test_space_invaders.py](file:///home/markliou/ML-practice/ReinforcementLearning/Atari/jax-1080ti/test_space_invaders.py)
- **用途**：導入功能模組，檢查環境是否正常運行並驗證其屬性。
- **特點**：
  - 印出 Action Space (含動作的意涵) 與 Observation Space 的格式。
  - 驗證 JAX Array 轉換是否成功，並顯示裝置（例如 `cpu:0`）。
  - 模擬隨機動作（5 個 Step），檢查步進時的 Reward、Done 與剩餘生命值。
- **執行測試方式**：
  ```bash
  docker run --rm -v /home/markliou/ML-practice:/workspace markliou/jax-1080ti python3 /workspace/ReinforcementLearning/Atari/jax-1080ti/test_space_invaders.py
  ```

---

## 3. 如何將環境導入您的訓練程式碼

您可以在您的訓練腳本中直接匯入並使用 `JaxAtariEnv`：

```python
from space_invaders import JaxAtariEnv

# 1. 建立環境
env = JaxAtariEnv()

# 2. 初始化環境，直接拿到 JAX Array 狀態
obs, info = env.reset()

# 3. 您的 RL 訓練迴圈
for step in range(total_steps):
    # 透過您的 PPO policy 預測動作
    action = policy(obs)  # policy 產出 JAX array 格式的 action
    
    # 執行動作，拿到的 obs、reward、done 全都是 JAX Array 格式！
    next_obs, reward, done, info = env.step(action)
    
    # ... 進行 PPO 緩衝區儲存與梯度更新 ...
    
    obs = next_obs
    if done:
        obs, info = env.reset()

env.close()
```
