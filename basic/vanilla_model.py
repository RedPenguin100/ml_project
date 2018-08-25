import tensorflow as tf
import numpy as np
import sys
from numpy import sqrt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from utils import *

SAVED_MODEL_PATH = "vanilla_model/"
np.set_printoptions(threshold=np.nan)

data = input_data.read_data_sets('MNIST-data', one_hot=True)

data.test.cls = np.array([label.argmax() for label in data.test.labels])
data.train.cls = np.array([label.argmax() for label in data.train.labels])

""" The actual model """
x = tf.placeholder(tf.float32, [None, img_size_flat], name="image")
y_true = tf.placeholder(tf.float32, [None, num_classes], name="y_true")
y_true_cls = tf.argmax(y_true, axis=1)

# First convolutional layer
x_image = tf.reshape(x, [-1, 28, 28, 1])

weights1 = get_weights([5, 5, 1, 32])
biases1 = get_bias([32])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, weights1, strides=[1, 1, 1, 1], padding='SAME') + biases1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Second convolutional layer

weights2 = get_weights([5, 5, 32, 64])
biases2 = get_bias([64])

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, weights2, strides=[1, 1, 1, 1], padding='SAME') + biases2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# First fully connected:
weights_fully_connected1 = get_weights([7 * 7 * 64, 1024])
biases_fully_connected1 = get_bias([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fully_connected1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights_fully_connected1) + biases_fully_connected1)

# Second fully connected
weights_fully_connected2 = get_weights([1024, num_classes])
biases_fully_connected2 = get_bias([num_classes])

h_fully_connected2 = tf.nn.relu(tf.matmul(h_fully_connected1, weights_fully_connected2) + biases_fully_connected2)

keep_probability = tf.placeholder(tf.float32)
fully_connected_drop = tf.nn.dropout(h_fully_connected2, keep_probability)

# Softmax!
y_pred = tf.nn.softmax(h_fully_connected2)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=h_fully_connected2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()

current_batch = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if len(sys.argv) != 2:
        raise ValueError("Wrong number of parameters")

    if sys.argv[1] == "train":
        for i in range(10000):
            current_batch += 50
            current_batch %= 55000
            batch_images = data.train.images[current_batch:current_batch + 50]
            batch_true = data.train.labels[current_batch:current_batch + 50]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_images, y_true: batch_true})
                print("step %d, training accuracy: %.4f" % (i, train_accuracy))
            if i % 1000 == 0:
                saver.save(sess=sess, save_path=SAVED_MODEL_PATH)
            optimizer.run(feed_dict={x: batch_images, y_true: batch_true, keep_probability: 0.5})
        saver.save(sess=sess, save_path=SAVED_MODEL_PATH)

    elif sys.argv[1] == "restore":
        saver.restore(sess=sess, save_path=SAVED_MODEL_PATH)
    else:
        raise ValueError("Bad parameter given.")

    def print_model_accuracy(image_array, true_label_array):
        image_count = len(image_array)
        if image_count != len(true_label_array):
            raise ValueError("Bad dimensions given")
        all_predictions = np.zeros(len(image_array))
        for j in range(image_count):
            all_predictions[j] = accuracy.eval(
                feed_dict={x: np.array(image_array[j:j + 1]),
                           y_true: np.array(true_label_array[j:j + 1]),
                           keep_probability: 1.0})
        correct_count = np.count_nonzero(all_predictions == 1)
        print("Accuracy is: %.4f" % (correct_count / image_count))

    print_model_accuracy(image_array=data.test.images, true_label_array=data.test.labels)