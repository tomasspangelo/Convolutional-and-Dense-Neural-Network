import numpy as np


class DataGenerator:
    def __init__(self, n, size_range, noise, train_size, test_size, val_size):
        self.n = n
        self.size_range = size_range
        self.noise = noise
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size

    def generate_data(self, size):
        return DataSet()


class DataSet:
    def __init__(self):
        self.data = []

    def add(self, image):
        self.data.append(image)

    def flatten(self):

        flat_data = []
        labels = []

        for image in self.data:
            flat_data.append(image.flatten())
            labels.append(image.get_label())

        return flat_data, labels


class Image:
    def __init__(self, flat, label):
        self.label = label
        self.flat = flat

    def flatten(self, vector=True):
        if vector:
            return np.reshape(self.flat, (-1,))

        return self.flat

    def get_label(self):
        return self.label

