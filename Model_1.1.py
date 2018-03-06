import tensorflow as tf
import numpy as np
from final import batch_generator_3
import matplotlib.pyplot as plt

# Placeholders for data & labels
images = tf.placeholder(tf.float32, shape=[None, 45, 225, 3])
labels = tf.placeholder(tf.float32, shape=[None, 5, 20])
lstm_in_maps = tf. placeholder(tf.float32, shape=[None, 5, 25, 8])

# Hyperparameter
epochs = 1
batch_size = 1
lr = 0.0001
length = 5


# We start by defining several 'layermethods'

# This method initializes the cell state and hidden state of our lstm
def init_states(annotations):
    with tf.variable_scope('initialization_layer'):
        mean = tf.reduce_mean(annotations, 3)

        cell_w = tf.get_variable('c_init_weight', [1, 25, 25], initializer=tf.truncated_normal_initializer(stddev=1))
        cell_b = tf.get_variable('c_init_bias', [25], initializer=tf.zeros_initializer())
        cell_init = tf.nn.tanh(tf.matmul(mean, cell_w) + cell_b)

        hidden_w = tf.get_variable('h_init_weight', [1, 25, 25], initializer=tf.truncated_normal_initializer(stddev=1))
        hidden_b = tf.get_variable('h_init_bias', [25], initializer=tf.zeros_initializer())
        hidden_init = tf.nn.tanh(tf.matmul(mean, hidden_w) + hidden_b)
        return cell_init, hidden_init


# Our soft attention model computes alpha and the context vector for every time step
def f_att(hidden_state, annotations):
    with tf.variable_scope('attention_layer'):
        w = tf.get_variable('a_weight', [1, 5, 8, 25], initializer=tf.truncated_normal_initializer(stddev=1))
        b = tf.get_variable('A_bias', [25], initializer=tf.zeros_initializer())
        e = tf.matmul(tf.matmul(tf.expand_dims(hidden_state, 2), annotations), w)+b
        alpha = tf.nn.softmax(tf.reshape(e, [1, 5, 25, 1]))
        context = annotations*alpha

        return context, alpha


# Here we define a really simple convolutional network to process our input images
def conv():
    with tf.variable_scope("Conv_Layer1"):
        bias = tf.get_variable("bias", [16], initializer=tf.zeros_initializer())
        kernels = tf.get_variable("kernel", [5, 5, 3, 16], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1))
        maps = tf.nn.tanh(tf.nn.conv2d(images, kernels, strides=[1, 1, 1, 1], padding='SAME') + bias)
        maps = tf.nn.max_pool(maps, [1, 2, 2, 1], strides=[1, 3, 3, 1], padding="SAME")

    with tf.variable_scope("Conv_Layer2"):
        bias = tf.get_variable("bias", [8], initializer=tf.zeros_initializer())
        kernels = tf.get_variable("kernel", [3, 3, 16, 8], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1))
        maps = tf.nn.tanh(tf.nn.conv2d(maps, kernels, strides=[1, 1, 1, 1], padding='SAME') + bias)
        maps = tf.nn.max_pool(maps, [1, 2, 2, 1], strides=[1, 3, 3, 1], padding="SAME")

        # The output is a feature map
        return maps


# Since we cut off our conv-Layers before training to obtain an earlier feature map, we have to train with the
# following method, which consists of a hidden, a dense layer and the training stuff
def train_conv(map, labels):
    with tf.variable_scope("hidden_layer", reuse=tf.AUTO_REUSE):
        flattened = tf.reshape(map, [-1, 1000])
        weights = tf.Variable(tf.truncated_normal([1000, 100], stddev=2048 ** (-1 / 2)))
        biases = tf.Variable(tf.constant(0.0, shape=[100]))
        hidden_layer = tf.nn.tanh(tf.matmul(flattened, weights) + biases)

    with tf.variable_scope("Output_layer", reuse=tf.AUTO_REUSE):
        weights = tf.Variable(tf.truncated_normal([100, 100], stddev=512 ** (-1 / 2)))
        biases = tf.Variable(tf.constant(0.0, shape=[100]))
        logits = tf.matmul(hidden_layer, weights) + biases

    with tf.variable_scope("conv_train"):

        flat_labels = tf.reshape(labels, [-1, 100])

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels, logits=logits)
        mean_cross_entropy = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(lr)
        training_step = optimizer.minimize(mean_cross_entropy)
        correct_prediction = tf.equal(tf.argmax(logits, 0), tf.argmax(flat_labels, 0))
        conv_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return conv_accuracy, training_step


# Our lstm cell gets updated every time step depending on its previous states, the last caption and the context vector,
# its' calculations are mainly taken directly from the paper
def lstm_forward(context, caption, hidden_state, cell_state):

    with tf.variable_scope("input_gate"):

        bias = tf.get_variable("bias", [25], initializer=tf.zeros_initializer())

        hidden_weights = tf.get_variable("hidden_weights", [1, 25, 25],
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        caption_weights = tf.get_variable("caption_weights", [1, 25, 25],
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", [1, 25, 25],
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        input = tf.sigmoid(tf.matmul(hidden_state, tf.reshape(hidden_weights, [-1, 25, 25]))
                           + tf.matmul(caption, tf.reshape(caption_weights, [-1, 25, 25]))
                           + tf.matmul(tf.reduce_mean(context, 3), tf.reshape(context_weights, [-1, 25, 25]))
                           + bias)

    with tf.variable_scope("forget_gate"):
        bias = tf.get_variable("bias", [25], initializer=tf.zeros_initializer())

        hidden_weights = tf.get_variable("hidden_weights", [1, 25, 25],
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        caption_weights = tf.get_variable("caption_weights", [1, 25, 25],
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", [1, 25, 25],
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        forget = tf.sigmoid(tf.matmul(hidden_state, hidden_weights)
                            + tf.matmul(caption, tf.reshape(caption_weights, [-1, 25, 25]))
                            + tf.matmul(tf.reduce_mean(context, 3), tf.reshape(context_weights, [-1, 25, 25]))
                            + bias)

    with tf.variable_scope("output_gate"):
        bias = tf.get_variable("bias", [25], initializer=tf.zeros_initializer())

        hidden_weights = tf.get_variable("hidden_weights", [1, 25, 25],
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        caption_weights = tf.get_variable("caption_weights", [1, 25, 25],
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", [1, 25, 25],
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        output = tf.sigmoid(tf.matmul(hidden_state, hidden_weights)
                            + tf.matmul(caption, tf.reshape(caption_weights, [-1, 25, 25]))
                            + tf.matmul(tf.reshape(tf.reduce_mean(context, 3), [-1, 5, 25]), context_weights)
                           + bias)

    with tf.variable_scope("output_gate"):
        bias = tf.get_variable("bias", [25], initializer=tf.zeros_initializer())

        hidden_weights = tf.get_variable("hidden_weights", [1, 25, 25],
                                         initializer=tf.truncated_normal_initializer(stddev=1))
        caption_weights = tf.get_variable("caption_weights", [1, 25, 25],
                                          initializer=tf.truncated_normal_initializer(stddev=1))
        context_weights = tf.get_variable("context_weights", [1, 25, 25],
                                          initializer=tf.truncated_normal_initializer(stddev=1))

        cell_state = (forget * cell_state) + input * tf.tanh(tf.matmul(hidden_state, hidden_weights)
                                 + tf.matmul(caption, caption_weights)
                                 + tf.matmul(tf.reduce_mean(context, 3), context_weights)
                                 + bias)

    with tf.variable_scope("update_hidden"):
        hidden_state = output * tf.tanh(cell_state)

    context, alpha = f_att(hidden_state, lstm_in_maps)

    # We have to return the non-variable states to remember their configurations
    return context, alpha, hidden_state, cell_state


# Here we decode the output of our lstm-cells for each time-step to get our labels
def decode(context, hidden_state):
    with tf.variable_scope("decoder"):
        w_c = tf.get_variable("weights", [1, 8, 20],
                              initializer=tf.truncated_normal_initializer(stddev=1))
        b_c = tf.get_variable("bias", [20], initializer=tf.zeros_initializer())

        w_h = tf.get_variable("hidden_weights", [1, 25, 5],
                              initializer=tf.truncated_normal_initializer(stddev=1))
        b_h = tf.get_variable("hidden_bias", [5], initializer=tf.zeros_initializer())

        context = tf.matmul(tf.reduce_mean(context, 2), w_c) + b_c
        hidden_state = tf.reduce_mean(tf.matmul(hidden_state, w_h), 2) + b_h
        logits = tf.matmul(tf.reshape(context, [20,5]), tf.reshape(hidden_state, [5,1]))

        return logits


# We train the lstm-cell with the decoded output and the class-labels
def train_lstm(decoded, labels):
    with tf.variable_scope("training_Layer"):
        decoded = tf.unstack(decoded, axis=0)
        labels = tf.unstack(labels, axis=1)

        for index, elem in enumerate(decoded):
            elem = tf.reshape(elem, [1, 20])

            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[index], logits=elem)
            loss = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.GradientDescentOptimizer(lr)
            training_step = optimizer.minimize(loss)
            correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(labels[index]), 1), tf.argmax(elem))
            lstm_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return lstm_accuracy, training_step


# In this part, we define the graph
# This variable serves as a 'makeshift embedding space' to train
caption = tf.get_variable("caption", [1, 5, 25], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))

# The Convolutional part of the network computes the input for the lstm
map = conv()

# ....and has to be trained, too
conv_acc, train_step = train_conv(map, labels)

# We initialize cell & hidden state, as well as the context-vector (attention is not really needed in the beginning)
c, h = init_states(map)
context, alpha = f_att(h, map)

# These lists are for storing the recursive output
context_list = []
alpha_list = []
caption_list = []

# This is the recursive call for the lstm-cell
with tf.variable_scope('run', reuse=tf.AUTO_REUSE):
    for x in range(length):
        context, alpha, h, c = lstm_forward(context, caption, h, c)
        cur_caption = decode(context, h)
        context_list.append(context)
        alpha_list.append(alpha)
        caption_list.append(cur_caption)
        lstm_acc, lstm_train_step = train_lstm(caption_list, labels)


# Here we start training, unfortunately we had big problems with this
conv_batch_gen = batch_generator_3.Attend()
batch_gen = batch_generator_3.Attend()

session = tf.Session()
session.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('.', session.graph)


conv_acc_list = []
lstm_acc_list = []
'''
for e in range(epochs):

    training_batch = conv_batch_gen.get_training_batch(batch_size)
    print('convolutional epoch {} started'.format(e))
    for steps in training_batch:
        try:
            training_data = steps
        except StopIteration:
            pass

        training_images = training_data[0]
        training_labels = training_data[1]

        _conv_acc, _ = session.run(
            [conv_acc, train_step],
            feed_dict={images: training_images, labels: training_labels})

        conv_acc_list.append(_conv_acc)


for i in range(epochs):

    training_batch = batch_gen.get_training_batch(batch_size)
    print('lstm epoch {} started'.format(i))
    for steps in training_batch:
        try:
            training_data = steps
        except StopIteration:
            pass

        training_images = training_data[0]
        training_labels = training_data[1]

        _lstm_acc, _ = session.run(
            [lstm_acc, lstm_train_step],
            feed_dict={images: training_images, labels: training_labels})

        lstm_acc_list.append(_lstm_acc)

print(conv_acc_list)

plt.plot(conv_acc_list)
plt.show()
'''
