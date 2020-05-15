import tensorflow as tf
from tensorflow.python.framework import graph_io

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Training Parameters
learning_rate = 0.001
num_steps = 50
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])


# Create model
def conv_net(x):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    cnn1 = tf.keras.layers.Conv2D(64, 3, strides=(1,1), activation=tf.nn.relu)(x)
    cnn2 = tf.keras.layers.Conv2D(64, 3, strides=(2,2), activation=tf.nn.relu)(cnn1)
    cnn3 = tf.keras.layers.Conv2D(128, 3, strides=(1,1), activation=tf.nn.relu)(cnn2)
    cnn4 = tf.keras.layers.Conv2D(128, 3, strides=(2,2), activation=tf.nn.relu)(cnn3)
    fc0 = tf.keras.layers.Flatten()(cnn4)
    fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(fc0)
    fc2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(fc1)
    out = tf.keras.layers.Dense(10)(fc2)
    return out
pass

# Construct model
logits = conv_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(tf.global_variables_initializer())

    # # get the node names
    # print([n.name for n in tf.get_default_graph().as_graph_def().node])
    # exit(0)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y
                                                                })
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    saver.save(sess, 'cnn')

    # try to free the model as .pb
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        ['dense_2/BiasAdd']
    )
    graph_io.write_graph(output_graph_def, './', 'cnn.pb', as_text=False)
    

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256]
                                      }))


