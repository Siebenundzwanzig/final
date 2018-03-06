import tensorflow as tf
import numpy as np
from final import batch_generator_3


images = tf.placeholder(tf.float32, shape=[None, 45, 225, 3])
labels = tf.placeholder(tf.float32, shape=[None, 5, 20])
epochs = 3
test_maps = tf.placeholder(tf.float32, shape=[None, 5, 25, 8])
batch_size = 1
size_in = [5, 25, 1]


class MODEL:

    def __init__(self, length=5):

        self._vocab = 20
        self._length = length
        self._single_width = 25
        self._single_height = 5
        self._dim_embed = self._dim_feature = 25

    def _init_states(self, annotations):
        with tf.variable_scope('initialization_layer'):
            mean = tf.reduce_mean(annotations, 3)

            cell_w = tf.get_variable('c_init_weight', [1, 25, 25], initializer=tf.truncated_normal_initializer(stddev=1))
            cell_b = tf.get_variable('c_init_bias', [25], initializer=tf.zeros_initializer())
            cell_init = tf.nn.tanh(tf.matmul(mean, cell_w) + cell_b)

            hidden_w = tf.get_variable('h_init_weight', [1, 25, 25], initializer=tf.truncated_normal_initializer(stddev=1))
            hidden_b = tf.get_variable('h_init_bias', [25], initializer=tf.zeros_initializer())
            hidden_init = tf.nn.tanh(tf.matmul(mean, hidden_w) + hidden_b)
            return cell_init, hidden_init

    def _f_att(self, hidden_state, annotations):
        with tf.variable_scope('attention_layer'):
            w = tf.get_variable('a_weight', [1, 5, 8, 25], initializer=tf.truncated_normal_initializer(stddev=1))
            b = tf.get_variable('A_bias', [25], initializer=tf.zeros_initializer())
            e = tf.matmul(tf.matmul(tf.expand_dims(hidden_state, 2), annotations), w)+b
            alpha = tf.nn.softmax(tf.reshape(e, [1, 5, 25, 1]))
            context = annotations*alpha

            return context, alpha

    def _conv(self):
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

            return maps


    def _train_conv(self, map, labels):
        with tf.variable_scope("Dense_Layer"):
            map = tf.reduce_mean(map, 0)
            dense = tf.reshape(map, [-1, 20, 5])
            print(dense)
            w = tf.get_variable("weights", [1, 20, 20], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))
            b = tf.get_variable("bias", [5], initializer=tf.zeros_initializer())
            drive = tf.matmul(w, dense)+b
            logits = tf.nn.softmax(drive)
            print(logits)

    def _lstm_forward(self, hidden_state, cell_state, context, caption):

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

        context, alpha = self._f_att(hidden_state, test_maps)

        return context, alpha, hidden_state

    def _decode(self, context, hidden_state):
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

    def _train_lstm(self, decoded, labels):
        with tf.variable_scope("training_Layer"):
            loss = 0
            decoded = tf.unstack(decoded, axis=0)
            labels = tf.unstack(labels, axis=1)

            for index, elem in enumerate(decoded):
                elem = tf.reshape(elem, [1, 20])
                #print(elem)
                #print(labels[index])
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[index], logits=elem)
                loss = tf.reduce_mean(cross_entropy)
                optimizer = tf.train.GradientDescentOptimizer(0.001)
                training_step = optimizer.minimize(loss)
                correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(labels[index]), 1), tf.argmax(elem))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #accuracy = tf.reduce_mean(accuracy)


#resized = tf.image.resize_images(map, [45, 225])
#print(resized)

caption = tf.get_variable("caption", [1, 5, 25], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1))

run = MODEL()

map = run._conv()
print(map)
run._train_conv(map, labels)


c, h = run._init_states(test_maps)
#print(c)

context, alpha = run._f_att(h, test_maps)

context_list = []
alpha_list = []
caption_list = []

with tf.variable_scope('run', reuse=tf.AUTO_REUSE):
    for x in range(run._length):
        context, alpha, hidden_state = run._lstm_forward(h, c, context, caption)
        cur_caption = run._decode(context, hidden_state)
        context_list.append(context)
        alpha_list.append(alpha)
        caption_list.append(cur_caption)
    run._train_lstm(caption_list, labels)

conv_batch_gen = batch_generator_3.Attend2()
batch_gen = batch_generator_3.Attend()

session = tf.Session()
session.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('.',
                                     session.graph)
'''
for e in range(epochs):

    conv_batch = conv_batch_gen.get_training_batch(6)

    for steps in conv_batch:
'''
'''
        with tf.variable_scope("conv_train"):
            logits = tf.unstack(logits, axis=0)
            print(logits)
            print(labels)
            for index, elem in enumerate(logits):
                elem = tf.reshape(elem, [5, 20])
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[index], logits=elem)
                loss = tf.reduce_mean(cross_entropy)
                optimizer = tf.train.GradientDescentOptimizer(0.001)
                training_step = optimizer.minimize(loss)
                #correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(labels[index]), 1), tf.argmax(elem))
                #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                # accuracy = tf.reduce_mean(accuracy)
'''

