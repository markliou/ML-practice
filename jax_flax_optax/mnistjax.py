import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from flax import nnx
import optax

# # 定義平均池化輔助函數
# def avg_pool(x, window_shape, strides):
#     window_shape_full = (1, window_shape[0], window_shape[1], 1)
#     strides_full = (1, strides[0], strides[1], 1)
#     pooled = jax.lax.reduce_window(x, 0.0, jax.lax.add, window_shape_full, strides_full, 'VALID')
#     return pooled / (window_shape[0] * window_shape[1])

# 使用 NNX 風格定義 CNN 模塊
# class CNN(nnx.Module):
#     def __init__(self, *, rngs: nnx.Rngs):
#         # MNIST 輸入是 28x28x1（灰度圖）
#         self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
#         self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
#         # 經過兩次 2x2 池化後，28x28 圖像變為 6x6
#         # 所以展平後的大小 = 6 * 6 * 64 = 2304
#         self.dense1 = nnx.Linear(in_features=3136, out_features=256, rngs=rngs)
#         self.dense2 = nnx.Linear(in_features=256, out_features=10, rngs=rngs)

#     def __call__(self, x):
#         x = self.conv1(x)
#         x = jax.nn.relu(x)
#         x = avg_pool(x, (2, 2), (2, 2))
#         x = self.conv2(x)
#         x = jax.nn.relu(x)
#         x = avg_pool(x, (2, 2), (2, 2))
#         x = x.reshape((x.shape[0], -1))  # 展平操作
#         x = self.dense1(x)
#         x = jax.nn.relu(x)
#         x = self.dense2(x)
#         return x

class CNN(nnx.Module):
    def __init__(self,
                 regularizer = 1e-4,
                 *args,
                 **kwargs):

        self.rngs = nnx.Rngs(3)
        self.regularizer = regularizer
        self.loss = 0.
        self.layers = []

        self.conv1 = nnx.Conv(1, 32, (3, 3), strides=2, padding='SAME', rngs=self.rngs) # [N, 16, 16,32]
        self.conv2 = nnx.Conv(32, 64, (3, 3), strides=2, padding='SAME', rngs=self.rngs) # [N, 8, 8, 64]
        self.conv3 = nnx.Conv(64, 128, (3, 3), strides=2, padding='SAME', rngs=self.rngs) # [N, 4, 4, 128]
        self.conv4 = nnx.Conv(128, 256, (3, 3), strides=2, padding='SAME', rngs=self.rngs) # [N, 2, 2, 256]
        self.conv5 = nnx.Conv(256, 10, (3, 3), strides=2, padding='SAME', rngs=self.rngs) # [N, 1, 1, 10]

        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

    def __call__(self, x):
        conv1 = nnx.relu(self.conv1(x))
        conv2 = nnx.relu(self.conv2(conv1))
        conv3 = nnx.relu(self.conv3(conv2))
        conv4 = nnx.relu(self.conv4(conv3))
        out = self.conv5(conv4)

        return out.reshape(-1, 10)

    # addtional loss
    def kernel_bias_L2regularization(self, conv):
        self.loss = 0
        for layer in self.layers:
            # weights regularization
            self.loss += (layer.kernel.value.sum() + layer.bias.value.sum()) ** 2 * self.regularizer
        return self.loss

# 加載並預處理 MNIST 數據集
ds_builder = tfds.builder('mnist')
ds_builder.download_and_prepare()
train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
train_images = jnp.float32(train_ds['image']) / 255.0  # 形狀：(60000, 28, 28, 1)
train_labels = jnp.array(train_ds['label'])            # 形狀：(60000,)
test_images = jnp.float32(test_ds['image']) / 255.0    # 形狀：(10000, 28, 28, 1)
test_labels = jnp.array(test_ds['label'])              # 形狀：(10000,)

# 初始化模型
params_key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(params=params_key)
model = CNN(rngs=rngs)

# 設置優化器
learning_rate = 1e-3
optim = optax.adam(learning_rate)
optimizer = nnx.Optimizer(model, optim)

@nnx.jit
def loss_fn(model, images, labels, ):
    logits = model(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss

# 定義訓練和評估函數
@nnx.jit
def train_step(model, optimizer, images, labels):
    loss, grads = nnx.value_and_grad(loss_fn)(model, images, labels)
    optimizer.update(grads)
    return loss

@nnx.jit
def evaluate(model, images, labels):
    logits = model(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = (jnp.argmax(logits, axis=1) == labels).mean()
    return loss, accuracy

# 訓練循環
num_epochs = 5
batch_size = 128
num_train = train_images.shape[0]

for epoch in range(1, num_epochs + 1):
    # 每個 epoch 打亂訓練數據
    perm = jax.random.permutation(jax.random.PRNGKey(epoch), num_train)
    perm_images = train_images[perm]
    perm_labels = train_labels[perm]

    # 批次訓練
    total_loss = 0.0
    num_batches = 0
    for i in range(0, num_train, batch_size):
        batch_images = perm_images[i:i+batch_size]
        batch_labels = perm_labels[i:i+batch_size]
        loss = train_step(model, optimizer, batch_images, batch_labels)
        total_loss += loss
        num_batches += 1

    # 在測試集上評估
    test_loss, test_accuracy = evaluate(model, test_images, test_labels)
    print(f"Epoch {epoch}: 訓練平均損失 = {float(total_loss/num_batches):.4f}, "
          f"測試損失 = {float(test_loss):.4f}, 測試準確率 = {float(test_accuracy)*100:.2f}%")
