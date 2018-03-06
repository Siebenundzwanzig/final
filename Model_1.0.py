import tensorflow as tf
import numpy as np
from final import batch_generator_3
#from final.dir import Dir
import matplotlib.pyplot as plt

# Hyperparameter
epochs = 3
batch_size = 8

# Placeholders
images = tf.placeholder(tf.float32, [None, 45, 225, 3])
labels = tf.placeholder(tf.float32, [None, 5, 20])

# default sizes for variables
input_size = [batch_size, 5, 25, 8]
weight_size = [batch_size, 5, 8, 8]


# Computes the context-vector with a soft attention model
def f_att(feature_map):

    with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        a_weights = tf.get_variable("attentions", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        att = tf.tanh(tf.matmul(feature_map, tf.reshape(a_weights, [-1, 5, 8, 8]))+bias)
        alpha = tf.nn.softmax(att)
        context = (alpha*feature_map)

        return context


# Percepton for cell- and hidden-state initialization
def f_init(annotation, reuse=tf.AUTO_REUSE):
    print(annotation)
    b1 = tf.get_variable("f_init_bias_1", [8], initializer=tf.zeros_initializer())
    w1 = tf.get_variable("f_init_weights_1", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))

    multi_1 = tf.matmul(annotation, tf.reshape(w1, [-1, 5, 8, 8]))+b1

    return multi_1


# This implementation of the lstm-cell calls upon itself recursively
# The functions are taken directly from the paper
def lstm(hidden_state, cell_state, embed_captions, context_vector, length):

    with tf.variable_scope("input_gate", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        caption_weights = tf.get_variable("caption_weights", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        hidden_weights = tf.get_variable("hidden_weights", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        input = tf.sigmoid(tf.matmul(embed_captions, tf.reshape(caption_weights, [-1, 5, 8, 8]))
                           + tf.matmul(hidden_state, tf.reshape(hidden_weights, [-1, 5, 8, 8]))
                           + tf.matmul(context_vector, tf.reshape(context_weights, [-1, 5, 8, 8]))
                           + bias)

    with tf.variable_scope("forget_gate", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        caption_weights = tf.get_variable("caption_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        hidden_weights = tf.get_variable("hidden_weights", weight_size,
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        forget = tf.sigmoid(tf.matmul(embed_captions, tf.reshape(caption_weights, [-1, 5, 8, 8]))
                            + tf.matmul(hidden_state, tf.reshape(hidden_weights, [-1, 5, 8, 8]))
                            + tf.matmul(context_vector, tf.reshape(context_weights, [-1, 5, 8, 8]))
                            + bias)

    with tf.variable_scope("output_gate", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        caption_weights = tf.get_variable("caption_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        hidden_weights = tf.get_variable("hidden_weights", weight_size,
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        output = tf.sigmoid(tf.matmul(embed_captions, tf.reshape(caption_weights, [-1, 5, 8, 8]))
                            + tf.matmul(hidden_state, tf.reshape(hidden_weights, [-1, 5, 8, 8]))
                            + tf.matmul(context_vector, tf.reshape(context_weights, [-1, 5, 8, 8]))
                            + bias)

    with tf.variable_scope("cell_state", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        caption_weights = tf.get_variable("caption_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        hidden_weights = tf.get_variable("hidden_weights", weight_size,
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", weight_size,
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        cell = (forget*cell_state)+input*tf.tanh(tf.matmul(embed_captions, tf.reshape(caption_weights, [-1, 5, 8, 8]))
                                                 + tf.matmul(hidden_state, tf.reshape(hidden_weights, [-1, 5, 8, 8]))
                                                 + tf.matmul(context_vector, tf.reshape(context_weights, [-1, 5, 8, 8]))
                                                 + bias)

    with tf.variable_scope("hidden_state", reuse=tf.AUTO_REUSE):
        hidden_state = output*tf.tanh(cell)

    # here we compute the current context-vector
    context_vector = f_att(context_vector)

    # the recursive call
    if length > 0:
        length = length-1
        lstm(hidden_state, cell_state, embed_captions, context_vector, length)

    # The decoder tries to decode the last output of the whole lstm (we tried to get a better implementation for this
    # in Model_1.1, since it actually should decode every time step
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [3], initializer=tf.zeros_initializer())
        kernels = tf.get_variable("kernel", [5, 5, 3, 8], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1))

        final_maps = tf.nn.tanh(tf.nn.conv2d_transpose(context_vector, kernels, [batch_size, 5, 25, 3], strides=[1, 1, 1, 1], padding='SAME') + bias)
        final_captions = tf.reshape(embed_captions, [-1, 20, 5, 10])
        caption_kernel = tf.get_variable("caption_kernel", [1, 1, 10, 1], initializer=tf.zeros_initializer())
        final_captions = tf.nn.conv2d(final_captions, caption_kernel, strides=[1, 1, 1, 1], padding="SAME")
        final_captions = tf.reshape(final_captions, [-1, 5, 20])

        return final_captions, final_maps


# Here we start the graph-construction by defining our convolutional layers
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

# This layer just calls upon the construction of the lstm-cell
with tf.variable_scope("start_lstm", reuse=tf.AUTO_REUSE):
    embed_captions = tf.get_variable("embedding_space", input_size, tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1))
    final_captions, final_maps = lstm(f_init(maps), f_init(maps), embed_captions, maps, 5)

# Our Training Layer backpropagates the Error through all layers (unlike in Model_1.1)
with tf.variable_scope("train"):
    print(labels)
    print(final_captions)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=final_captions)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('mean_cross_entropy', mean_cross_entropy)
    optimizer = tf.train.AdamOptimizer(0.01)
    training_step = optimizer.minimize(mean_cross_entropy)

    preds = tf.argmax(final_captions, 2)
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(final_captions), 2), tf.argmax(labels, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 1)
    accuracy = tf.reduce_mean(accuracy)
    tf.summary.scalar('accuracy', accuracy)


# These are lists for storing the accuracy and loss
loss_list = []
train_acc_list = []
val_acc_list = []

batch_gen = batch_generator_3.Attend()

session = tf.Session()
merged = tf.summary.merge_all()

# train_writer = tf.summary.FileWriter('./train', session.graph)
# test_writer = tf.summary.FileWriter('./test')

# What follows now is standard training, validation and a single test
# Sadly we do not see any increase in accuracy, no matter the Hyperparameter
session.run(tf.global_variables_initializer())

e = 0
for x in range(epochs):

    training_batch = batch_gen.get_training_batch(8)

    print('epoch {} started'.format(e))
    i = 0
    for steps in training_batch:
        i += 1
        try:
            training_data = steps
        except StopIteration:
            pass
        training_images = training_data[0]
        training_labels = training_data[1]

        _summaries, train_maps, loss, _, train_accuracy = session.run(
            [merged, final_maps, mean_cross_entropy, training_step, accuracy],
            feed_dict={images: training_images, labels: training_labels})

        #train_writer.add_summary(_summaries)

        train_acc_list.append(train_accuracy)

        if i % 50 == 0:

            validation_batch = batch_gen.get_validation_batch(8)
            validation_data = next(validation_batch)
            validation_images = validation_data[0]
            validation_labels = validation_data[1]

            _v_summaries, val_maps, val_loss, val_accuracy = session.run([merged, final_maps, mean_cross_entropy, accuracy],
                                                           feed_dict={images: validation_images, labels: validation_labels})

            #test_writer.add_summary(_v_summaries)
            val_acc_list.append(val_accuracy)


test_batch = batch_gen.get_test_batch(8)
test_data = next(test_batch)
test_images = test_data[0]
test_labels = test_data[1]

out_map, test_loss, test_acc = session.run([final_maps, mean_cross_entropy, accuracy],
                                           feed_dict={images: test_images, labels: test_labels})





