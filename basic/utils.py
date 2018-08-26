import logging
import tensorflow as tf
import numpy as np
import sys
from numpy import sqrt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

img_size = 28
img_size_flat = img_size * img_size
num_classes = 10
deterministic_seed = 7
tf.set_random_seed(deterministic_seed)
accuracy_log = "accuracy_log.txt"


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
