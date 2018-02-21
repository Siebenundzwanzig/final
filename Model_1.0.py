import tensorflow as tf
import numpy as np


# Hyperparameter
L = 10


images = tf.placeholder(tf.float32, [None, 225, 45, 3])
labels = tf.placeholder(tf.int32, [None])


with tf.variable_scope("Conv Layer 1"):
    bias = tf.get_variable("bias", [16], initializer=tf.zeros_initializer())
    kernels = tf.get_variable("kernel", [5, 5, 3, 16], tf.float32, tf.truncated_normal_initializer(stddev=1))
    maps = tf.nn.tanh(tf.nn.conv2d(images, kernels, strides=[1, 1, 1, 1], padding='SAME')+bias)


#
dims = np.asarray(maps.get_shape())
print(dims)
annotations = tf.placeholder(tf.float32, [L, int(int(dims[1])/L), int(int(dims[2])/L), 3])
print(annotations)

#def LSTM(hidden_state, captions, context_vector):

