import os
import shutil
import tensorflow as tf
import numpy as np
import sys
from numpy import sqrt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import confusion_matrix

MODEL_NAME = os.path.basename(os.path.splitext(__file__)[0])
logger = get_logger(MODEL_NAME)
SAVED_MODEL_PATH = "{}/".format(MODEL_NAME)
tf.set_random_seed(deterministic_seed)
np.set_printoptions(threshold=np.nan)


def convert_label_to_simplex_point(label, num_classes=10):
    """
    Function assumes labels go from 0 to num_classes - 1
    """
    n = num_classes - 1
    if label == n:
        return np.ones(n) * (1 + sqrt(1 + n)) / n
    vec = np.zeros(n)
    vec[label] = 1
    return vec


def simplex_to_one_hot(point, num_classes=10):
    a = np.zeros(num_classes)
    if 1.0 not in point:
        a[num_classes - 1] = 1
    else:
        a[np.argmax(point)] = 1
    return a


def get_closest_simplex_point(point):
    n = len(point)
    entire_simplex = np.array([convert_label_to_simplex_point(label) for label in range(n + 1)])
    differences = []
    for simplex_point in entire_simplex:
        differences.append(np.linalg.norm(simplex_point - point))
    index = np.argmin(differences)
    return entire_simplex[index]


data = input_data.read_data_sets('MNIST-data', one_hot=True, seed=deterministic_seed)

data.train.simplex = np.array([convert_label_to_simplex_point(label.argmax()) for label in data.train.labels])
data.test.simplex = np.array([convert_label_to_simplex_point(label.argmax()) for label in data.test.labels])
data.test.cls = np.array([label.argmax() for label in data.test.labels])
data.train.cls = np.array([label.argmax() for label in data.train.labels])

x = tf.placeholder(tf.float32, [None, img_size_flat], name="image")
y_true = tf.placeholder(tf.float32, [None, num_classes], name="y_true")
y_true_simplex = tf.placeholder(tf.float32, [None, num_classes - 1], name="y_simplex")
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

# h_pool2 is now "picture" of size 7 * 7
# First fully connected layer
weights_fully_connected1 = get_weights([7 * 7 * 64, 1024])
biases_fully_connected1 = get_bias([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fully_connected1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights_fully_connected1) + biases_fully_connected1)

# Second fully connected
weights_fully_connected2 = get_weights([1024, 1024])
biases_fully_connected2 = get_bias([1024])

h_fully_connected2 = tf.matmul(h_fully_connected1, weights_fully_connected2) + biases_fully_connected2

# Third fully connected layer
weights_fully_connected3 = get_weights([1024, num_classes - 1])
biases_fully_connected3 = get_bias([num_classes - 1])

h_fully_connected3 = tf.matmul(h_fully_connected2, weights_fully_connected3) + biases_fully_connected3

# Finally, simplex!
y_conv = h_fully_connected3
distance = tf.reduce_sum(tf.norm(y_true_simplex - y_conv, ord='euclidean', axis=1))

optimizer = tf.train.AdamOptimizer(1e-4).minimize(distance)

entire_simplex = np.array([convert_label_to_simplex_point(label) for label in range(num_classes)])
entire_tf_simplex = tf.constant(entire_simplex, dtype=tf.float32)
differences = tf.map_fn(lambda point: tf.norm(point - entire_tf_simplex, ord='euclidean', axis=1), y_conv)
closest_point_index = tf.argmin(differences, 1)
closest_point_on_simplex_to_y_conv = tf.map_fn(lambda single_index: entire_tf_simplex[single_index],
                                               closest_point_index, dtype=tf.float32)
correct_prediction = tf.equal(y_true_simplex, closest_point_on_simplex_to_y_conv)
correct_prediction = tf.map_fn(lambda truth: tf.reduce_all(truth), correct_prediction)
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
            batch_simplex = data.train.simplex[current_batch:current_batch + 50]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_images, y_true_simplex: batch_simplex})
                print("step %d, training accuracy: %.4f" % (i, train_accuracy))
            if i % 1000 == 0:
                saver.save(sess=sess, save_path=SAVED_MODEL_PATH)
            optimizer.run(feed_dict={x: batch_images, y_true_simplex: batch_simplex})
        saver.save(sess=sess, save_path=SAVED_MODEL_PATH)

    elif sys.argv[1] == "extract_layers":
        saver.restore(sess=sess, save_path=SAVED_MODEL_PATH)

        flatten_h_pool1 = tf.reshape(h_pool1, shape=[-1, 14 * 14 * 32])
        flatten_h_pool2 = tf.reshape(h_pool2, shape=[-1, 7 * 7 * 64])
        h_fully_connected1 = tf.reshape(h_fully_connected1, shape=[-1, 1024])
        h_fully_connected2 = tf.reshape(h_fully_connected2, shape=[-1, 1024])
        h_fully_connected3 = tf.reshape(h_fully_connected3, shape=[-1, 9])
        batch_size = 200
        layer_file_names = ["simplex_3variation_1.txt", "simplex_3variation_2.txt", "simplex_3variation_3.txt",
                            "simplex_3variation_4.txt", "simplex_3variation_5.txt"]
        file_handles = [open(layer_filename, "a") for layer_filename in layer_file_names]
        for i in range(len(data.train.images) // batch_size):
            dict_to_feed = {x: data.train.images[batch_size * i: batch_size * i + batch_size],
                            y_true: data.train.labels[batch_size * i: batch_size * i + batch_size]}

            np.savetxt(file_handles[0], flatten_h_pool1.eval(feed_dict=dict_to_feed), fmt="%f")
            np.savetxt(file_handles[1], flatten_h_pool2.eval(feed_dict=dict_to_feed), fmt="%f")
            np.savetxt(file_handles[2], h_fully_connected1.eval(feed_dict=dict_to_feed), fmt="%f")
            np.savetxt(file_handles[3], h_fully_connected2.eval(feed_dict=dict_to_feed), fmt="%f")
            np.savetxt(file_handles[4], h_fully_connected3.eval(feed_dict=dict_to_feed), fmt="%f")
            if i % 10 == 0:
                print("Current iteration: {}".format(i))
        [file.close() for file in file_handles]
    elif sys.argv[1] == "restore":
        saver.restore(sess=sess, save_path=SAVED_MODEL_PATH)
        all_points = np.zeros(len(data.test.images))
        batch = 100
        for j in range(len(all_points) // batch):
            tmp = np.array(data.test.simplex[batch * j:batch * j + batch])
            all_points[batch * j: batch * j + batch] = closest_point_index.eval(
                feed_dict={x: np.array(data.test.images[batch * j:batch * j + batch]),
                           y_true_simplex: tmp})
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
    else:
        raise ValueError("Bad parameter given.")
