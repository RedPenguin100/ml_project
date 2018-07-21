import tensorflow as tf
import numpy as np
from numpy import sqrt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)


def plot_images(images, cls_true, cls_pred=None, img_shape=(28, 28)):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {}".format(cls_true[i])
        else:
            xlabel = "True: {}, Pred: {}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def get_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def get_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


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
    entire_simplex = np.array([convert_one_hot_to_simplex_point(label) for label in range(n+1)])
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
img_size = 28
img_size_flat = img_size * img_size
num_classes = 10

x = tf.placeholder(tf.float32, [None, img_size_flat], name="image")
y_true = tf.placeholder(tf.float32, [None, num_classes], name="y_true")
y_simplex = tf.placeholder(tf.float32, [None, num_classes - 1], name="y_simplex")
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
weights_fully_connected = get_weights([7 * 7 * 64, 1024])
biases_fully_connected = get_bias([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fully_connected = tf.nn.relu(tf.matmul(h_pool2_flat, weights_fully_connected) + biases_fully_connected)

keep_prob = tf.placeholder(tf.float32)
h_fully_connected_drop = tf.nn.dropout(h_fully_connected, keep_prob)

# Finally, softmax!
y_conv = tf.matmul(h_fully_connected, get_weights([1024, num_classes - 1])) + get_bias([num_classes - 1])
print("Before printing")
print(y_conv)
print(y_simplex)
distance = tf.reduce_sum(tf.norm(y_simplex - y_conv, ord='euclidean', axis=1))

train_step = tf.train.AdamOptimizer(1e-4).minimize(distance)

entire_simplex = np.array([convert_one_hot_to_simplex_point(label) for label in range(num_classes)])
entire_tf_simplex = tf.constant(entire_simplex, dtype=tf.float32)
print(entire_tf_simplex.shape)
# point - 9 dimensions
# entire tf_simplex - 10 * 9
differences = tf.map_fn(
    lambda point: tf.norm(point - entire_tf_simplex, ord='euclidean', axis=1), y_conv)
print(differences)
closest_point_index = tf.argmin(differences, 1)
closest_point_on_simplex_to_y_conv = tf.map_fn(lambda single_index: entire_tf_simplex[single_index],
                                               closest_point_index, dtype=tf.float32)
# closest_point_on_simplex_to_y_conv =
print("Closest Point shape", closest_point_on_simplex_to_y_conv.shape)
closest_point_on_simplex_to_y_conv = tf.Print(closest_point_on_simplex_to_y_conv, [closest_point_on_simplex_to_y_conv],
                                              summarize=100, message="Closest point")
# y_simplex = tf.Print(y_simplex, [y_simplex], message="y_conv")
correct_prediction = tf.equal(y_simplex, closest_point_on_simplex_to_y_conv)
correct_prediction = tf.map_fn(lambda truth: tf.reduce_all(truth), correct_prediction)
print(correct_prediction.shape)
correct_prediction = tf.Print(correct_prediction, [correct_prediction], message="Correct prediction", summarize=1000)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

current_batch = 0
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        current_batch += 50
        current_batch %= 55000
        batch_images = data.train.images[current_batch:current_batch + 50]
        batch_simplex = data.train.simplex[current_batch:current_batch + 50]
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_images, y_simplex: batch_simplex, keep_prob: 1.0})
            print("step %d, training accuracy %.4f" % (i, train_accuracy))
            # for z in range(10):
                # print("Labels:{}".format(data.train.cls[current_batch + z]))
        train_step.run(feed_dict={x: batch_images, y_simplex: batch_simplex, keep_prob: 0.5})

    test_sample = 2000
    res = sess.run(y_conv, feed_dict={x: data.test.images[0:test_sample], y_simplex: data.test.simplex[0:test_sample]})
    counter = 0
    prediction_labels = []
    for i, point in enumerate(res):
        print("Original point: {}".format(point))
        print("Feed: {}".format(data.test.simplex[i]))
        point = get_closest_simplex_point(point)
        one_hot = simplex_to_one_hot(point)
        prediction_labels.append(np.argmax(one_hot))
        print("Prediction:{}, True:{}".format(np.argmax(one_hot), data.test.cls[i]))
        if one_hot.argmax() == data.test.cls[i]:
            counter += 1
    prediction_labels = np.array(prediction_labels)
    temp = data.test.cls[0:2000]
    correct = prediction_labels == temp
    print(len(correct))
    print ("Was correct {} times, out of {}".format(counter, test_sample))

    print("test accuracy %.4f" % accuracy.eval(feed_dict={
        x: data.test.images[0:2000], y_simplex: data.test.simplex[0:2000], keep_prob: 1.0}))

    print("test accuracy %.4f" % accuracy.eval(feed_dict={
        x: data.test.images[2000:4000], y_simplex: data.test.simplex[2000:4000], keep_prob: 1.0}))

    print("test accuracy %.4f" % accuracy.eval(feed_dict={
        x: data.test.images[4000:6000], y_simplex: data.test.simplex[4000:6000], keep_prob: 1.0}))

    print("test accuracy %.4f" % accuracy.eval(feed_dict={
        x: data.test.images[6000:8000], y_simplex: data.test.simplex[6000:8000], keep_prob: 1.0}))


    def plot_example_errors(cls_pred, correct):
        # This function is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # correct is a boolean array whether the predicted class
        # is equal to the true class for each image in the test-set.

        # Negate the boolean array.
        incorrect = (correct == False)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = data.test.images[0:2000][incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = data.test.cls[0:2000][incorrect]

        # Plot the first 9 images.
        plot_images(images=images[0:9],
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])

    plot_example_errors(prediction_labels, correct)