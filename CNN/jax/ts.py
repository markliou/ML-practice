import jax
from jax import random
import numpy as np
from jax.scipy.special import logsumexp
from jax.experimental import stax, optimizers
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax)

key = jax.random.PRNGKey(1)

init_fun, conv_net = stax.serial(Conv(32, (5, 5), (2, 2), padding="SAME"),
                                 BatchNorm(), Relu,
                                 Conv(32, (5, 5), (2, 2), padding="SAME"),
                                 BatchNorm(), Relu,
                                 Conv(10, (3, 3), (2, 2), padding="SAME"),
                                 BatchNorm(), Relu,
                                 Conv(10, (3, 3), (2, 2), padding="SAME"), Relu,
                                 Flatten,
                                 Dense(10),
                                 LogSoftmax)

batch_size = 32
_, params = init_fun(key, (batch_size, 1, 28, 28))

step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)
num_epochs = 10

# train_loss, train_log, test_log = run_mnist_training_loop(num_epochs,
#                                                           opt_state,
#                                                           net_type="CNN")