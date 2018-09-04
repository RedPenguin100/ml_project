import os
import logging
import time
import tensorflow as tf
import numpy as np
import sys
import shutil
from numpy import sqrt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import confusion_matrix

MODEL_NAME = os.path.basename(os.path.splitext(__file__)[0])
logger = get_logger(MODEL_NAME)

SAVED_MODEL_PATH = "{}/".format(MODEL_NAME)
np.set_printoptions(threshold=np.nan)
tf.set_random_seed(deterministic_seed)
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

h_fully_connected2 = tf.matmul(h_fully_connected1, weights_fully_connected2) + biases_fully_connected2

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
number_of_iterations = 10000
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if len(sys.argv) != 2:
        raise ValueError("Wrong number of parameters")

    if sys.argv[1] == "train":
        shutil.rmtree(SAVED_MODEL_PATH, ignore_errors=True)
        for i in range(number_of_iterations):
            current_batch += 50
            current_batch %= 55000
            batch_images = data.train.images[current_batch:current_batch + 50]
            batch_true = data.train.labels[current_batch:current_batch + 50]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_images, y_true: batch_true})
                print("step %d, training accuracy: %.4f" % (i, train_accuracy))
            if i % 1000 == 0:
                saver.save(sess=sess, save_path=SAVED_MODEL_PATH)
            optimizer.run(feed_dict={x: batch_images, y_true: batch_true})
        saver.save(sess=sess, save_path=SAVED_MODEL_PATH)

    elif sys.argv[1] == "extract_layers":
        saver.restore(sess=sess, save_path=SAVED_MODEL_PATH)

        flatten_h_pool1 = tf.reshape(h_pool1, shape=[-1, 14 * 14 * 32])
        flatten_h_pool2 = tf.reshape(h_pool2, shape=[-1, 7 * 7 * 64])
        h_fully_connected1 = tf.reshape(h_fully_connected1, shape=[-1, 1024])
        h_fully_connected2 = tf.reshape(h_fully_connected2, shape=[-1, 10])
        batch_size = 200
        layer_file_names = ["vanilla_layer_1.txt", "vanilla_layer_2.txt", "vanilla_layer_3.txt", "vanilla_layer_4.txt"]
        file_handles = [open(layer_filename, "a") for layer_filename in layer_file_names]
        for i in range(len(data.train.images) // batch_size):
            dict_to_feed = {x: data.train.images[batch_size * i: batch_size * i + batch_size],
                            y_true: data.train.labels[batch_size * i: batch_size * i + batch_size]}

            np.savetxt(file_handles[0], flatten_h_pool1.eval(feed_dict=dict_to_feed), fmt="%f")
            np.savetxt(file_handles[1], flatten_h_pool2.eval(feed_dict=dict_to_feed), fmt="%f")
            np.savetxt(file_handles[2], h_fully_connected1.eval(feed_dict=dict_to_feed), fmt="%f")
            np.savetxt(file_handles[3], h_fully_connected2.eval(feed_dict=dict_to_feed), fmt="%f")
            if i % 10 == 0:
                print("Current iteration: {}".format(i))
        [file.close() for file in file_handles]
    elif sys.argv[1] == "restore":
        saver.restore(sess=sess, save_path=SAVED_MODEL_PATH)

        all_points = np.zeros(len(data.test.images))
        batch = 100
        for j in range(len(all_points) // batch):
            tmp = np.array(data.test.labels[batch * j:batch * j + batch])
            all_points[batch * j: batch * j + batch] = y_pred_cls.eval(
                feed_dict={x: np.array(data.test.images[batch * j:batch * j + batch]),
                           y_true: tmp})
        cm = confusion_matrix(y_true=data.test.cls, y_pred=all_points)

        all_precisions = []
        all_recalls = []
        for i in range(num_classes):
            precision = get_precision(i, cm)
            recall = get_recall(i, cm)
            all_precisions.append(precision)
            all_recalls.append(recall)
            print("Precision for {}: %.3f".format(i) % precision)
            print("Recall for {}: %.3f".format(i) % recall)
        print("Model accuracy: {}".format(get_model_accuracy(all_points, data.test.cls)))
        correct = all_points == data.test.cls
        plot_confusion_matrix_nicely(cm)
    else:
        raise ValueError("Bad parameter given.")

