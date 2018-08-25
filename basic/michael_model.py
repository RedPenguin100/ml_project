import tensorflow as tf
import numpy as np
import sys
from numpy import sqrt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from utils import plot_images, get_weights, get_bias, img_size, img_size_flat, num_classes

SAVED_MODEL_PATH = "michael_model/"
np.set_printoptions(threshold=np.nan)


def convert_one_hot_to_simplex_point(label, num_classes=10):
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
    entire_simplex = np.array([convert_one_hot_to_simplex_point(label) for label in range(n + 1)])
    differences = []
    for simplex_point in entire_simplex:
        differences.append(np.linalg.norm(simplex_point - point))
    index = np.argmin(differences)
    return entire_simplex[index]


data = input_data.read_data_sets('MNIST-data', one_hot=True)

data.train.simplex = np.array([convert_one_hot_to_simplex_point(label.argmax()) for label in data.train.labels])
data.test.simplex = np.array([convert_one_hot_to_simplex_point(label.argmax()) for label in data.test.labels])
data.test.cls = np.array([label.argmax() for label in data.test.labels])
data.train.cls = np.array([label.argmax() for label in data.train.labels])
# print(data.test.cls)

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

# For model 2
# # Third convolutional layer
# weights3 = get_weights([5, 5, 64, 64])
# biases3 = get_bias([64])

# h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, weights3, strides=[1, 1, 1, 1], padding='SAME') + biases3)

# h_pool2 is now "picture" of size 7 * 7
# First fully connected layer
weights_fully_connected1 = get_weights([7 * 7 * 64, 1024])
biases_fully_connected1 = get_bias([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fully_connected1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights_fully_connected1) + biases_fully_connected1)

# Second fully connected layer
weights_fully_connected2 = get_weights([1024, num_classes - 1])
biases_fully_connected2 = get_bias([num_classes - 1])

h_fully_connected2 = tf.matmul(h_fully_connected1, weights_fully_connected2) + biases_fully_connected2

keep_prob = tf.placeholder(tf.float32)
h_fully_connected_drop = tf.nn.dropout(h_fully_connected2, keep_prob)

# Finally, simplex!
y_conv = h_fully_connected2
distance = tf.reduce_sum(tf.norm(y_true_simplex - y_conv, ord='euclidean', axis=1))

train_step = tf.train.AdamOptimizer(1e-4).minimize(distance)

entire_simplex = np.array([convert_one_hot_to_simplex_point(label) for label in range(num_classes)])
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
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if len(sys.argv) != 2:
        raise ValueError("Wrong number of parameters")
    if sys.argv[1] == "train":
        for i in range(10000):
            current_batch += 50
            current_batch %= 55000
            batch_images = data.train.images[current_batch:current_batch + 50]
            batch_simplex = data.train.simplex[current_batch:current_batch + 50]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch_images, y_true_simplex: batch_simplex, keep_prob: 1.0})
                print("step %d, training accuracy %.4f" % (i, train_accuracy))
            if i % 1000 == 0:
                saver.save(sess=sess, save_path=SAVED_MODEL_PATH)
            train_step.run(feed_dict={x: batch_images, y_true_simplex: batch_simplex, keep_prob: 0.5})

        saver.save(sess=sess, save_path=SAVED_MODEL_PATH)
    elif sys.argv[1] == "restore":
        saver.restore(sess=sess, save_path=SAVED_MODEL_PATH)
    else:
        raise ValueError("Bad parameter given.")


    def print_model_accuracy(image_array, simplex_label_array):
        image_count = len(image_array)
        if image_count != len(simplex_label_array):
            raise ValueError("Bad dimensions given")
        all_predictions = np.zeros(len(image_array))
        for j in range(image_count):
            all_predictions[j] = accuracy.eval(
                feed_dict={x: np.array(image_array[j:j + 1]),
                           y_true_simplex: np.array(simplex_label_array[j:j + 1]),
                           keep_prob: 1.0})
        correct_count = np.count_nonzero(all_predictions == 1)
        print("Accuracy is: %.4f" % (correct_count / image_count))


    print_model_accuracy(image_array=data.test.images, simplex_label_array=data.test.simplex)
