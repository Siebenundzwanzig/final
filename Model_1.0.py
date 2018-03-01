import tensorflow as tf
import numpy as np
from final import batch_generator_3
#from final.dir import Dir
import matplotlib.pyplot as plt
# Hyperparameter
epochs = 1


time_steps = 5
#Number of annotations
L = 10
vocab_size = 20
#vocab = tf.placeholder(tf.int32, [None])#[vocab_size]?


images = tf.placeholder(tf.float32, [None, 45, 225, 3])
labels = tf.placeholder(tf.int32, [None, 5, 20])
batch_size = 8
#print(batch_size)
input_size = [batch_size, 5, 25, 8]
weight_size = [batch_size, 5, 8, 8]
#embed_captions = tf.nn.embedding_lookup(captions, labels)


def f_att(feature_map):

    with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        a_weights = tf.get_variable("attentions", weight_size, initializer=tf.truncated_normal_initializer(stddev=1))
        #print(feature_map)
        #print(a_weights)
        att = tf.tanh(tf.matmul(feature_map, tf.reshape(a_weights, [-1, 5, 8, 8]))+bias)
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

    context_vector = f_att(context_vector)

    if length > 0:
        #print(length)
        length = length-1
        lstm(hidden_state, cell_state, embed_captions, context_vector, length)

    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [3], initializer=tf.zeros_initializer())
        kernels = tf.get_variable("kernel", [5, 5, 3, 8], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1))
        #print(context_vector)
        #print(final_maps)

        final_maps = tf.nn.tanh(tf.nn.conv2d_transpose(context_vector, kernels, [batch_size, 5, 25, 3], strides=[1, 1, 1, 1], padding='SAME') + bias)
        #final_maps = tf.nn.conv2d_transpose(final_maps, kernels [])
        #final_maps = tf.reshape(final_maps, [-1, 5, 25, 3])
        #print(final_maps)
        final_captions = tf.reshape(embed_captions, [-1, 20, 5, 10])
        caption_kernel = tf.get_variable("caption_kernel", [1, 1, 10, 1], initializer=tf.zeros_initializer())
        final_captions = tf.nn.conv2d(final_captions, caption_kernel, strides=[1, 1, 1, 1], padding="SAME")
        final_captions = tf.reshape(final_captions, [-1, 5, 20])
        return final_captions, final_maps


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
#print(maps)
with tf.variable_scope("start_lstm"):
    embed_captions = tf.get_variable("embedding_space", input_size, tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1))
    #print(embed_captions)
    final_captions, final_maps = lstm(maps, maps, embed_captions, maps, 5)
    #print(final_maps)

with tf.variable_scope("train"):
    #print(final_captions)
    #print(final_maps)
    #print(labels)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=final_captions)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(0.0001)
    training_step = optimizer.minimize(mean_cross_entropy)
    #print(final_captions)
    #print(labels)
    correct_prediction = tf.equal(tf.argmax(final_captions, 0), tf.argmax(labels, 0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss_list = []
train_acc_list = []
val_acc_list = []

batch_gen = batch_generator_3.Attend()
#print(batch_gen.get_sizes())
session = tf.Session()
session.run(tf.global_variables_initializer())

for x in range(epochs):

    training_batch = batch_gen.get_training_batch(8)

    print('epoch {} started'.format(epochs))
    i = 0
    for steps in training_batch:
        #print(i)
        i += 1
        try:
            training_data = steps
        except StopIteration:
            pass
        training_images = training_data[0]
        #print(training_images.shape)
        training_labels = training_data[1]
        #print(training_labels.shape)
        #print(training_labels.shape)
        #training_images = np.resize(training_images, [batch_size, 225, 45, 3])
        #print(training_labels.shape)

        train_maps, loss, _, train_accuracy = session.run([final_maps, mean_cross_entropy, training_step, accuracy],
                                              feed_dict={images: training_images, labels: training_labels})


        train_acc_list.append(train_accuracy)

        if i % 50 == 0:

            validation_batch = batch_gen.get_validation_batch(8)
            validation_data = next(validation_batch)
            validation_images = validation_data[0]
            validation_labels = validation_data[1]

            val_maps, val_loss, val_accuracy = session.run([final_maps, mean_cross_entropy, accuracy],
                                                           feed_dict={images: validation_images, labels: validation_labels})

            val_acc_list.append(val_accuracy)

test_batch = batch_gen.get_test_batch(8)
test_data = next(test_batch)
test_images = test_data[0]
test_labels = test_data[1]

out_map, test_loss, test_acc = session.run([final_maps, mean_cross_entropy, accuracy],
                                           feed_dict={images: test_images, labels: test_labels})


print(train_acc_list)
print(val_acc_list)

print(out_map.shape)
print(test_images.shape)

map1 = np.asarray(out_map)[1]
map1 = np.absolute(map1)
print(map1)
fig = plt.figure()
plt.imshow(map1)
plt.show()


# attentions, final_maps, final_captions

