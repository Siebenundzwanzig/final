import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
class ATTEND():
    def __init__(self, directory = '/home/laars/uni/WS2017/tensorflow/final/glued_images'):
        self._directory = directory

        self._training_data = []
        self._training_labels = []
        self._validation_data = []
        self._validation_labels = []
        self._test_data = []
        self._test_labels = []

        self._load_traing_data()
        self._load_validation_data()
        self._load_test_data()

        # np.random.seed(0)
        # samples_n = self._training_labels.shape[0]
        # random_indices = np.random.choice(samples_n, samples_n // 10, replace = False)
        # np.random.seed()
        #
        # self._validation_data = self._training_data[random_indices]
        # self._validation_labels = self._training_labels[random_indices]
        # self._training_data = np.delete(self._training_data, random_indices, axis = 0)
        # self._training_labels = np.delete(self._training_labels, random_indices)


    def _load_traing_data(self):
        for _, _, files in os.walk(self._directory + "training"):
            for image in files:
                print(image)
                self._training_data.append(np.asarray(Image.open(self._directory + "training" + image)))
                self._training_labels.append(np.array(image.replace(".jpg", "")))

    def _load_validation_data(self):
        for _, _, files in os.walk(self._directory + "validation"):
            for image in files:
                print(image)
                self._training_data.append(np.asarray(Image.open(self._directory + "validation" + image)))
                self._training_labels.append(np.array(image.replace(".jpg", "")))


    def _load_test_data(self):
        for _, _, files in os.walk(self._directory + "testing"):
            for image in files:
                print(image)
                self._training_data.append(np.asarray(Image.open(self._directory + "testing" + image)))
                self._training_labels.append(np.array(image.replace(".jpg", "")))

    def get_training_batch(self, batch_size):
        return self._get_batch(self._training_data, self._training_labels, batch_size)

    def get_validation_batch(self, batch_size):
        return self._get_batch(self._validation_data, self._validation_labels, batch_size)

    def get_test_batch(self, batch_size):
        return self._get_batch(self._test_data, self._test_labels, batch_size)

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


    def get_sizes(self):
        training_samples_n = self._training_labels.shape[0]
        validation_samples_n = self._validation_labels.shape[0]
        test_samples_n = self._test_labels.shape[0]
        return training_samples_n, validation_samples_n, test_samples_n
