import tensorflow as tf
import numpy as np


# Hyperparameter
time_steps = 5
#Number of annotations
L = 10

input_size = [1, 25, 5, 8]
weight_size = [1, 25, 8, 8]
images = tf.placeholder(tf.float32, [None, 225, 45, 3])
labels = tf.placeholder(tf.int32, [None])
captions = tf.get_variable("embedding_space", input_size, tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))


def f_att(feature_map):

    with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        a_weights = tf.get_variable("attentions", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        att = tf.tanh(tf.matmul(feature_map, a_weights)+bias)
        alpha = tf.nn.softmax(att)
        context = (alpha*feature_map)
        return context


def lstm(hidden_state, cell_state, captions, context_vector, length):

    print(hidden_state, cell_state, captions, context_vector, length)

    with tf.variable_scope("input_gate", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        caption_weights = tf.get_variable("caption_weights", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        hidden_weights = tf.get_variable("hidden_weights", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        input = tf.sigmoid(tf.matmul(captions, caption_weights)
                           + tf.matmul(hidden_state, hidden_weights)
                           + tf.matmul(context_vector, context_weights)
                           + bias)

    with tf.variable_scope("forget_gate", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        caption_weights = tf.get_variable("caption_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        hidden_weights = tf.get_variable("hidden_weights", weight_size,
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        forget = tf.sigmoid(tf.matmul(captions, caption_weights)
                           + tf.matmul(hidden_state, hidden_weights)
                           + tf.matmul(context_vector, context_weights)
                           + bias)

    with tf.variable_scope("output_gate", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        caption_weights = tf.get_variable("caption_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        hidden_weights = tf.get_variable("hidden_weights", weight_size,
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        output = tf.sigmoid(tf.matmul(captions, caption_weights)
                           + tf.matmul(hidden_state, hidden_weights)
                           + tf.matmul(context_vector, context_weights)
                           + bias)

    with tf.variable_scope("cell_state", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        caption_weights = tf.get_variable("caption_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        hidden_weights = tf.get_variable("hidden_weights", weight_size,
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        cell = (forget*cell_state)+input*tf.tanh(tf.matmul(captions, caption_weights)+ tf.matmul(hidden_state, hidden_weights)+ tf.matmul(context_vector, context_weights)+ bias)

    with tf.variable_scope("hidden_state", reuse=tf.AUTO_REUSE):
        hidden_state = output*tf.tanh(cell)

    context_vector = f_att(context_vector)

    if length >= 0:
        print(length)
        length = length-1
        lstm(hidden_state, cell_state, captions, context_vector, length)


with tf.variable_scope("ConvLayer1"):
    bias = tf.get_variable("bias", [16], initializer=tf.zeros_initializer())
    kernels = tf.get_variable("kernel", [5, 5, 3, 16], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))
    maps = tf.nn.tanh(tf.nn.conv2d(images, kernels, strides=[1, 1, 1, 1], padding='SAME')+bias)
    maps = tf.nn.max_pool(maps, [1, 2, 2, 1], strides=[1, 3, 3, 1], padding="SAME")
#print(maps)

with tf.variable_scope("ConvLayer2"):
    bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
    kernels = tf.get_variable("kernel", [3, 3, 16, 8], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))
    maps = tf.nn.tanh(tf.nn.conv2d(maps, kernels, strides=[1, 1, 1, 1], padding='SAME')+bias)
    maps = tf.nn.max_pool(maps, [1, 2, 2, 1], strides=[1, 3, 3, 1], padding="SAME")
print(maps)

"""
with tf.variable_scope("ConvLayer1"):
    bias = tf.get_variable("bias", [16], initializer=tf.zeros_initializer())
    kernels = tf.get_variable("kernel", [5, 5, 3, 16], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))
    maps = tf.nn.tanh(tf.nn.conv2d(images, kernels, strides=[1, 1, 1, 1], padding='SAME')+bias)

with tf.variable_scope("ConvLayer1"):
    bias = tf.get_variable("bias", [16], initializer=tf.zeros_initializer())
    kernels = tf.get_variable("kernel", [5, 5, 3, 16], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))
    maps = tf.nn.tanh(tf.nn.conv2d(images, kernels, strides=[1, 1, 1, 1], padding='SAME')+bias)

with tf.variable_scope("ConvLayer1"):
    bias = tf.get_variable("bias", [16], initializer=tf.zeros_initializer())
    kernels = tf.get_variable("kernel", [5, 5, 3, 16], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))
    maps = tf.nn.tanh(tf.nn.conv2d(images, kernels, strides=[1, 1, 1, 1], padding='SAME')+bias)
"""


lstm(maps, maps, captions, maps, 5)
#print(f_att(maps))
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./summaries/train')


#test_writer = tf.summary.FileWriter('/test')

#k = f_att(maps)
#print(k)


#Ideas for annotation-vector generation
#dims = np.asarray(maps.get_shape())
#print(dims)
#annotations = tf.placeholder(tf.float32, [L, int(int(dims[1])/L), int(int(dims[2])/L), 3])
#alpha = tf.placeholder(tf.float32, [L, int(int(dims[1])/L), int(int(dims[2])/L), 1])
#print(annotations)





'''
   

    (hidden_state, cell_state, captions, context_vector, length-1)

'''