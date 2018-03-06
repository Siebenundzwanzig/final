# Generator for batches
# Very similar to the CIFAR batch generator of second homework.
# Redesigned for our purposes.
# Thank you, Lukas ;)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class Attend():
    # Initialize with given directory, load numpy files as training, validation
    # and test data and labels
    def __init__(self, directory = '/home/laars/uni/WS2017/tensorflow/final/glued_images/'):
        self._directory = directory
        self._training_data = np.load(directory + 'training_data.npy')
        self._training_labels = np.load(directory + 'training_labels.npy')
        self._validation_data = np.load(directory + 'validation_data.npy')
        self._validation_labels = np.load(directory + 'validation_labels.npy')
        self._test_data = np.load(directory + 'testing_data.npy')
        self._test_labels = np.load(directory + 'testing_labels.npy')

    def get_training_batch(self, batch_size):
        return self._get_batch(self._training_data, self._training_labels, batch_size)

    def get_validation_batch(self, batch_size):
        return self._get_batch(self._validation_data, self._validation_labels, batch_size)

    def get_test_batch(self, batch_size):
        return self._get_batch(self._test_data, self._test_labels, batch_size)

    # Returns a generator with all data and labels in random order and
    # as many as wished.
    def _get_batch(self, data, labels, batch_size):
        samples_n = labels.shape[0]
        if batch_size <= 0:
            batch_size = samples_n

        random_indices = np.random.choice(samples_n, samples_n, replace = False)
        data = data[random_indices]
        labels = labels[random_indices]
        for i in range(samples_n // batch_size):
            on = i * batch_size
            off = on + batch_size
            yield data[on:off], labels[on:off]

    # returns number of data
    def get_sizes(self):
        training_samples_n = self._training_labels.shape[0]
        validation_samples_n = self._validation_labels.shape[0]
        test_samples_n = self._test_labels.shape[0]
        return training_samples_n, validation_samples_n, test_samples_n

# If called as main, show with matplotlib.pyplot.
# Used for debugging
if __name__ == "__main__":
    attend = Attend()
    sample = attend.get_training_batch(5)
    print(attend.get_sizes())
    fig = plt.figure()
    img = next(sample)
    for i in range(5):
        label=img[1][i]
        ax=fig.add_subplot(150+i+1)
        ax.set_title("label: " + str(label))
        imgplot = ax.imshow(img[0][i])
    plt.show()
