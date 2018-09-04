import logging
import string

import tensorflow as tf
import numpy as np
import sys
from numpy import sqrt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

img_size = 28
img_size_flat = img_size * img_size
num_classes = 10
deterministic_seed = 7
tf.set_random_seed(deterministic_seed)
accuracy_log = "accuracy_log.txt"


def plot_images(images, cls_true, cls_pred=None, img_shape=(28, 28)):
    assert len(images) == len(cls_true) == 9, "images: {}, cls_true: {}".format(len(images), len(cls_true))

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


def plot_example_errors(cls_pred, correct, data):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect].astype(int)

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def get_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, seed=deterministic_seed))


def get_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def get_logger(name):
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    file_handler = logging.FileHandler(accuracy_log)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def get_fp(label, conf_matrix):
    fp = 0
    n = len(conf_matrix)
    for i in range(n):
        if i == label:
            continue
        fp += conf_matrix[i][label]
    return fp


def get_fn(label, conf_matrix):
    fn = 0
    n = len(conf_matrix)
    for i in range(n):
        if i == label:
            continue
        fn += conf_matrix[label][i]
    return fn


def get_tn(label, conf_matrix):
    return conf_matrix.trace() - get_tp(label, conf_matrix)


def get_tp(label, conf_matrix):
    return conf_matrix[label][label]


def get_precision(label, conf_matrix):
    return get_tp(label, conf_matrix) / (get_tp(label, conf_matrix) + get_fp(label, conf_matrix))


def get_recall(label, conf_matrix):
    return get_tp(label, conf_matrix) / (get_tp(label, conf_matrix) + get_fn(label, conf_matrix))


def get_model_accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Bad dimensions given")
    return np.sum(y_true == y_pred) / len(y_pred)


def plot_confusion_matrix_nicely(conf_matrix, dataset='MNIST'):
    if dataset == 'MNIST':
        df_cm = pd.DataFrame(conf_matrix, index=[i for i in string.digits], columns=[i for i in string.digits]).astype('int64')
    else:
        raise NotImplementedError("Not yet implemented for dataset: {}".format(dataset))
    plt.ticklabel_format(style='plain')
    plt.figure(figsize=(10,7))

    sn.heatmap(df_cm, annot=True, fmt="d")

    plt.show()