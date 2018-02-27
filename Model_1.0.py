import tensorflow as tf
import numpy as np
from final import batch_generator_3
#from final.dir import Dir

# Hyperparameter
epochs = 2
batch_size = 4


time_steps = 5
#Number of annotations
L = 10
vocab_size = 20
#vocab = tf.placeholder(tf.int32, [None])#[vocab_size]?

input_size = [1, 25, 5, 8]
weight_size = [1, 25, 8, 8]
images = tf.placeholder(tf.float32, [None, 225, 45, 3])
labels = tf.placeholder(tf.int32, [None, 5])
captions = tf.get_variable("embedding_space", input_size, tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))
embed_captions = tf.nn.embedding_lookup(captions, vocab_size)


def f_att(feature_map):

    with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        a_weights = tf.get_variable("attentions", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        att = tf.tanh(tf.matmul(feature_map, a_weights)+bias)
        alpha = tf.nn.softmax(att)
        context = (alpha*feature_map)
        return context

#TODO initialization of hidden and cell
def lstm(hidden_state, cell_state, embed_captions, context_vector, length):

    with tf.variable_scope("input_gate", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        caption_weights = tf.get_variable("caption_weights", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        hidden_weights = tf.get_variable("hidden_weights", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        input = tf.sigmoid(tf.matmul(embed_captions, caption_weights)
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
        forget = tf.sigmoid(tf.matmul(embed_captions, caption_weights)
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
        output = tf.sigmoid(tf.matmul(embed_captions, caption_weights)
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
        cell = (forget*cell_state)+input*tf.tanh(tf.matmul(embed_captions, caption_weights)+tf.matmul(hidden_state, hidden_weights)+tf.matmul(context_vector, context_weights)+bias)

    with tf.variable_scope("hidden_state", reuse=tf.AUTO_REUSE):
        hidden_state = output*tf.tanh(cell)

    context_vector = f_att(context_vector)

    if length >= 0:
        print(length)
        length = length-1
        lstm(hidden_state, cell_state, embed_captions, context_vector, length)

    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [3], initializer=tf.zeros_initializer())
        kernels = tf.get_variable("kernel", [5, 5, 3, 8], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1))
        global final_maps
        final_maps = tf.nn.tanh(tf.nn.conv2d_transpose(context_vector, kernels, [1, 225, 45, 3], strides=[1, 1, 1, 1], padding='SAME') + bias)
        global final_captions
        final_captions = tf.reshape(embed_captions, [-1, 5, 20])


with tf.variable_scope("ConvLayer1"):
    bias = tf.get_variable("bias", [16], initializer=tf.zeros_initializer())
    kernels = tf.get_variable("kernel", [5, 5, 3, 16], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))
    maps = tf.nn.tanh(tf.nn.conv2d(images, kernels, strides=[1, 1, 1, 1], padding='SAME')+bias)
    maps = tf.nn.max_pool(maps, [1, 2, 2, 1], strides=[1, 3, 3, 1], padding="SAME")


with tf.variable_scope("ConvLayer2"):
    bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
    kernels = tf.get_variable("kernel", [3, 3, 16, 8], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))
    maps = tf.nn.tanh(tf.nn.conv2d(maps, kernels, strides=[1, 1, 1, 1], padding='SAME')+bias)
    maps = tf.nn.max_pool(maps, [1, 2, 2, 1], strides=[1, 3, 3, 1], padding="SAME")


with tf.variable_scope("start_lstm"):
    lstm(maps, maps, embed_captions, maps, 5)


with tf.variable_scope("train"):
    print(final_captions)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=final_captions)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(0.0001)
    training_step = optimizer.minimize(mean_cross_entropy)
    print(final_captions)
    print(labels)
    correct_prediction = tf.equal(tf.argmax(final_captions, 2), tf.argmax(tf.one_hot(labels, 5), 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss_list = []
train_acc_list = []


batch_gen = batch_generator_3.Attend()
session = tf.session()
session.run(tf.global_variables_initializer())

for x in range(epochs):

    training_batch = batch_gen.get_training_batch(batch_size)

    print('epoch {} started'.format(epochs))

    for steps in training_batch:
        training_data = next(training_batch)

        training_images = training_data[0]
        training_labels = training_data[1]

        loss, _, train_accuracy = session.run([mean_cross_entropy, training_step, accuracy],
                                              feed_dict={images: training_images,
                                                           labels: training_labels})

        train_acc_list.append(train_accuracy)

print(train_acc_list)
#merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter('./summaries/train')


