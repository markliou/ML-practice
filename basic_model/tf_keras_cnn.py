import tensorflow as tf 

def cnn(x, reuse=True):
    # Input : 416 * 416
    with tf.variable_scope('cnn', reuse=reuse):
        x = tf.image.resize_images(x, (416, 416))
        conv1 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation=tf.nn.relu, padding="SAME")(x) #208
        conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation=tf.nn.relu, padding="SAME")(conv1) #104
        conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation=tf.nn.relu, padding="SAME")(conv2) #52
        conv4 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation=tf.nn.relu, padding="SAME")(conv3) #26
        conv5 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation=tf.nn.relu, padding="SAME")(conv4) #13
        
        fin = tf.keras.layers.Flatten()(conv5)
        f1 = tf.keras.layers.Dense(512, activation=tf.nn.tanh)(fin)
        f2 = tf.keras.layers.Dense(256, activation=tf.nn.tanh)(f1)
        f3 = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(f2)
        
        out = tf.keras.layers.Dense(2, activation=tf.tanh)(f3)
    return out
    
pass