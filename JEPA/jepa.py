import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import copy
import random
import math

# --- 1. 超參數設定 (Hyperparameters) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 50 # 為了展示，設定少一點。真實訓練需要更多。
LEARNING_RATE = 1e-4
MOMENTUM = 0.996 # Target Encoder EMA 更新的動量係數

# --- 2. 輔助工具 (Utilities) ---

class EMAUpdate:
    """
    處理 Target Encoder 的動量平均更新
    """
    def __init__(self, momentum):
        self.momentum = momentum

    def __call__(self, online_net, target_net):
        with torch.no_grad():
            for online_params, target_params in zip(online_net.parameters(), target_net.parameters()):
                target_params.data = self.momentum * target_params.data + (1.0 - self.momentum) * online_params.data

def get_cifar10_dataloader(batch_size):
    """
    載入 CIFAR-10 數據
    """
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloader

# --- 3. 模型架構 (Model Architecture) ---

class CNNEncoder(nn.Module):
    """
    一個簡單的 CNN 編碼器
    Input: (B, 3, 32, 32)
    Output: (B, 256, 4, 4) -> 扁平化 -> (B, 256 * 4 * 4)
    """
    def __init__(self, in_channels=3, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2) # 4x4
        )
        self.out_dim = out_dim * 4 * 4 # 256 * 4 * 4 = 4096

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

class Predictor(nn.Module):
    """
    一個簡單的 MLP 預測器
    Input: Context feature + Target position token
    Output: Predicted target feature
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, context_feature, pos_embedding):
        x = torch.cat([context_feature, pos_embedding], dim=1)
        return self.net(x)

class JEPA(nn.Module):
    """
    組合所有元件的 JEPA 主模型
    """
    def __init__(self, encoder_dim=4096, predictor_hidden_dim=512, pos_embed_dim=128):
        super().__init__()
        self.context_encoder = CNNEncoder()
        # Target Encoder 一開始是 Context Encoder 的複製品，且不計算梯度
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # 每個 Target Patch 有一個可學習的位置編碼
        # 假設我們最多有 4 個 target blocks
        self.num_targets = 4
        self.pos_embeddings = nn.Parameter(torch.randn(self.num_targets, pos_embed_dim))

        self.predictor = Predictor(input_dim=encoder_dim + pos_embed_dim,
                                   hidden_dim=predictor_hidden_dim,
                                   output_dim=encoder_dim)

    def generate_masks(self, x, num_targets=4, max_scale=0.8, min_scale=0.4):
        """
        為一個 batch 的圖片產生 context 和 target 遮罩
        返回: context_mask (B, 1, H, W), target_masks (B, N, 1, H, W)
        """
        # 使用傳入的 x 的 shape 來決定 Batch size 和 H, W
        B, _, H, W = x.shape

        context_mask = torch.zeros(B, 1, H, W, device=x.device)
        target_masks = torch.zeros(B, num_targets, 1, H, W, device=x.device)

        for i in range(B):
            c_h = int(H * random.uniform(min_scale, max_scale))
            c_w = int(W * random.uniform(min_scale, max_scale))
            c_y = random.randint(0, H - c_h)
            c_x = random.randint(0, W - c_w)
            context_mask[i, :, c_y:c_y+c_h, c_x:c_x+c_w] = 1

            for j in range(num_targets):
                t_h = int(c_h * random.uniform(0.25, 0.5))
                t_w = int(c_w * random.uniform(0.25, 0.5))
                t_y = random.randint(c_y, c_y + c_h - t_h)
                t_x = random.randint(c_x, c_x + c_w - t_w)
                target_masks[i, j, :, t_y:t_y+t_h, t_x:t_x+t_w] = 1

        return context_mask, target_masks

    def forward(self, x):
        # 1. 產生遮罩
        context_mask, target_masks = self.generate_masks(x, num_targets=self.num_targets)

        # 2. 準備輸入
        context_view = x * context_mask
        target_views = x.unsqueeze(1) * target_masks # (B, N, C, H, W)

        # 3. 取得 Context Feature
        context_feature = self.context_encoder(context_view) # (B, D)

        # 4. 取得 Target Features (無梯度)
        with torch.no_grad():
            B, N, C, H, W = target_views.shape
            target_features = self.target_encoder(target_views.view(B*N, C, H, W)) # (B*N, D)
            target_features = target_features.view(B, N, -1) # (B, N, D)

        # 5. 進行預測並計算 Loss
        total_loss = 0
        for i in range(self.num_targets):
            # 取得對應的位置編碼並擴展到整個 batch
            pos_emb = self.pos_embeddings[i].unsqueeze(0).expand(B, -1) # (B, P_D)

            # 進行預測
            predicted_feature = self.predictor(context_feature, pos_emb) # (B, D)

            # 計算 L2 Loss
            loss = F.mse_loss(predicted_feature, target_features[:, i, :])
            total_loss += loss

        return total_loss / self.num_targets


# --- 4. 訓練迴圈 (Training Loop) ---

def main():
    print(f"使用裝置: {DEVICE}")

    # 1. 準備模型和數據
    model = JEPA().to(DEVICE)
    dataloader = get_cifar10_dataloader(BATCH_SIZE)

    # 只需要優化 context_encoder 和 predictor
    optimizer = torch.optim.Adam(list(model.context_encoder.parameters()) + list(model.predictor.parameters()), lr=LEARNING_RATE)

    # 初始化 EMA 更新器
    ema_updater = EMAUpdate(MOMENTUM)

    # 2. 開始訓練
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (images, _) in enumerate(dataloader):
            images = images.to(DEVICE)

            # 計算 loss
            loss = model(images)

            # 優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新 Target Encoder
            ema_updater(model.context_encoder, model.target_encoder)

            total_loss += loss.item()

            if (i+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch [{epoch+1}/{EPOCHS}] 完成, 平均 Loss: {avg_loss:.4f} ---")

    print("訓練完成！")
    # 此處可以保存 context_encoder 的權重，用於下游任務
    # torch.save(model.context_encoder.state_dict(), 'jepa_cnn_cifar10.pth')

if __name__ == '__main__':
    main()
