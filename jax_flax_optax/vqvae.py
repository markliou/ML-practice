import tensorflow_datasets as tfds
import numpy as np
import jax
import flax.nnx
import jax.numpy as jnp
import tensorflow as tf
import optax
from PIL import Image

# create the dataset
batchSize = 128
dsimg = tfds.load("beans", split='train', shuffle_files=True, batch_size=-1)['image'].numpy()
reImg = tf.image.resize(dsimg, [256,256])
dataset = tf.data.Dataset.from_tensor_slices(reImg)
# dataset = dataset.batch(batchSize, drop_remainder=True).repeat().shuffle(3, reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
dataset = dataset.repeat().shuffle(3, reshuffle_each_iteration=True).batch(batchSize, drop_remainder=True).map(lambda x: tf.image.random_crop(value=x, size=(batchSize, 128, 128, 3)), num_parallel_calls=8).prefetch(8)
ds_iter = iter(dataset)

class vqvae(flax.nnx.Module):
    def __init__(self,
                 codebookSize = 256,
                 l2reg = 1e-4,
                 *args, **kwargs):
        super(flax.nnx.Module, self).__init__(*args, **kwargs)

        self.rngs = flax.nnx.Rngs(0)
        self.codebookSize = codebookSize
        self.activation = flax.nnx.swish
        self.l2reg = l2reg
        self.layers = []


        # input size: [-1, 128, 128, 3]
        # encoder
        self.conv1 = flax.nnx.Conv(3, 32, (11,11), strides=2, rngs=self.rngs) # [-1, 64, 64, 32]
        self.layers.append(self.conv1)
        self.conv1c = flax.nnx.Conv(32, 32, (3,3), strides=1, rngs=self.rngs) # [-1, 64, 64, 32]
        self.layers.append(self.conv1c)

        self.conv2 = flax.nnx.Conv(32, 64, (11,11), strides=2, rngs=self.rngs) # [-1, 32, 32, 64]
        self.layers.append(self.conv2)
        self.conv2c = flax.nnx.Conv(64, 64, (3,3), strides=1, rngs=self.rngs) # [-1, 32, 32, 64]
        self.layers.append(self.conv2c)

        self.conv3 = flax.nnx.Conv(64, 128, (11,11), strides=2, rngs=self.rngs) # [-1, 16, 16, 128]
        self.layers.append(self.conv3)
        self.conv3c = flax.nnx.Conv(128, 128, (3,3), strides=1, rngs=self.rngs) # [-1, 16, 16, 128]
        self.layers.append(self.conv3c)

        # decoder
        self.tconv1 = flax.nnx.ConvTranspose(128, 64, (5,5), strides=2, rngs=self.rngs) # [-1, 32, 32, 64]
        self.layers.append(self.tconv1)
        self.tconv1c = flax.nnx.Conv(64, 64, (3,3), strides=1, rngs=self.rngs) # [-1, 32, 32, 64]
        self.layers.append(self.conv2c)

        self.tconv2 = flax.nnx.ConvTranspose(64, 32, (5,5), strides=2, rngs=self.rngs) # [-1, 64, 64, 32]
        self.layers.append(self.tconv2)
        self.tconv2c = flax.nnx.Conv(32, 32, (3,3), strides=1, rngs=self.rngs) # [-1, 64, 64, 32]
        self.layers.append(self.tconv2c)

        self.tconv3 = flax.nnx.ConvTranspose(32, 32, (5,5), strides=2, rngs=self.rngs) # [-1, 128, 128, 32]
        self.layers.append(self.tconv3)
        self.tconv3c = flax.nnx.Conv(32, 3, (3,3), strides=1, rngs=self.rngs) # [-1, 64, 64, 3]
        self.layers.append(self.tconv3c)

        # setting code book
        self.codeBook = flax.nnx.Param(
            jax.nn.initializers.orthogonal()(jax.random.key(0), (1, self.codebookSize, 128))
            # jax.nn.initializers.truncated_normal(0.01)(jax.random.key(0), (1, self.codebookSize, 64))
            )

        # setting the monitored variable
        self.candidateLatents = flax.nnx.Variable(jnp.zeros((1,)))

    # @flax.nnx.jit
    def __call__(self, input):

        # encoding
        d1 = self.activation(self.conv1(input))
        d1 = self.activation(self.conv1c(d1))
        d2 = self.activation(self.conv2(d1))
        d2 = self.activation(self.conv2c(d2))
        d3 = self.activation(self.conv3(d2))
        d3 = self.conv3c(d3)

        # reshaping for code exchange
        candidateLatents = jnp.reshape(d3, [-1, 1, 128])
        self.candidateLatents.value = candidateLatents
        # calculating distances
        euDis = jnp.sum((candidateLatents - self.codeBook) ** 2, axis = -1) # [-1, 128]
        activeIndex = jnp.argmin(euDis, axis=-1) # [-1]
        # replacing codes
        activeIndexOnehot = jax.nn.one_hot(activeIndex, self.codebookSize) # [-1, 128]
        self.sow(flax.nnx.Intermediate, "activeIndexOnehot", activeIndexOnehot, reduce_fn=lambda x, y: y)
        replacedLatents = jnp.sum(
            jnp.reshape(activeIndexOnehot, [-1, self.codebookSize, 1]) * self.codeBook,
            axis = -2
        ) # [-1, 128]

        # reshaping replaced latents
        replacedLatents = jnp.reshape(replacedLatents, d3.shape)
        # straight throught estimated
        replacedLatents = jax.lax.stop_gradient(replacedLatents - d3) + d3

        d4 = self.activation(self.tconv1(replacedLatents))
        d4 = self.activation(self.tconv1c(d4))
        d5 = self.activation(self.tconv2(d4))
        d5 = self.activation(self.tconv2c(d5))
        d6 = self.activation(self.tconv3(d5))
        out = self.tconv3c(d6)

        return out

    def l2Reg(self):
        regLoss = 0.
        for layer in self.layers:
            regLoss += jnp.sum(jax.tree_util.tree_leaves(layer.kernel)[0] ** 2)
            regLoss += jnp.sum(jax.tree_util.tree_leaves(layer.bias)[0] ** 2)
        return regLoss * self.l2reg

    def commit_and_vq_loss(self):
        replacedLatents = jnp.sum(
            ## !! use nnx.Param.value object for gradient. Or Jax will use nnx.Parameter.view that only give value without computation graph edges !! ##
            jnp.reshape(self.activeIndexOnehot, [-1, self.codebookSize, 1]) * self.codeBook.value,
            axis = -2
        ) # [-1, 128]
        candidateLatens4Loss = jnp.reshape(self.candidateLatents, [-1, 128])
        commitLoss = jnp.mean((jax.lax.stop_gradient(candidateLatens4Loss) - replacedLatents) ** 2)  # commit loss
        vqLoss = jnp.mean((candidateLatens4Loss - jax.lax.stop_gradient(replacedLatents)) ** 2) *.25 # vq loss
        self.sow(flax.nnx.Intermediate, "commitLoss", commitLoss, reduce_fn=lambda x, y: y)
        self.sow(flax.nnx.Intermediate, "vqLoss", vqLoss, reduce_fn=lambda x, y: y)
        return commitLoss + vqLoss


model = vqvae()

@flax.nnx.jit
def loss_fn(model, x):
    y = (jnp.array(x) / 255. ) - 1.
    y_hat = model(y)
    se = jnp.mean((y_hat - y) ** 2)
    return (se + model.commit_and_vq_loss() + model.l2Reg())

learningRate = 1e-4

optChain = optax.chain(
   optax.clip_by_global_norm(1.0),
   optax.adamw(learningRate),
)
opt = flax.nnx.Optimizer(model, optChain)
grad_fn = flax.nnx.value_and_grad(loss_fn)

# @flax.nnx.jit # cause abnormal
def update_model_weights(model, y):
   loss, grads = grad_fn(model, y)
   opt.update(grads)
   return loss

trainingStep = 500000
for step in range(trainingStep):
    x = jnp.array(next(ds_iter))
    loss = update_model_weights(model, x)
    if step % 100 == 0 :
        print("step:{}  loss:{}".format(step, loss))

        y = (jnp.array(x) / 255. ) - 1.
        y_hat = model(y)

        def give_img(x, name):
            pic = x
            # pic = tf.reshape(pic, [128,128,3])
            # pic = tf.reshape(pic, [3,128,128])
            # pic = tf.transpose(pic, [1,2,0])
            pic = (pic + 1) * 128
            pic = Image.fromarray(tf.cast(pic, tf.uint8).numpy())
            pic.save(name)

        give_img(y[0], 'bean.jpg')
        give_img(y_hat[0], 'bean_hat.jpg')
