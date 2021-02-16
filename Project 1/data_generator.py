import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:
    """
    DataGenerator object.
    """

    @staticmethod
    def generate_data(n, height_range, width_range, noise, size, num_classes=4, center=True):
        """
        Static method for generating data given image defining parameters.
        :param n: Size of the nxn background of the image
        :param height_range: Tuple indicating (min, max) height of object in image
        :param width_range: Tuple indicating (min, max) width of object in image
        :param noise: Floating number between 0 and 1 indicating amount of noise.
        :param size: Size of the dataset to be generated.
        :param num_classes: The number of classes. 1, 2, 3 or 4.
        :param center: True if object should be centered in image, False otherwise.
        :return: DataSet object.
        """

        dataset = DataSet()
        per_class = int(size / num_classes)
        rest_class = int(size % num_classes)

        for c in range(num_classes):
            for _ in range(per_class):
                height = np.random.randint(height_range[0], height_range[1] + 1)
                width = np.random.randint(width_range[0], width_range[1] + 1)
                base = DataGenerator.create_base(Image.get_figure_type(c), height, width)

                figure_type = Image.get_figure_type(c)
                if np.random.uniform() < 0.7 and figure_type != "vertical bars":
                    base.rotate_90()
                base.add_background(n, center)
                base.add_noise(noise)
                dataset.add(base)

        for r in range(rest_class):
            height = np.random.randint(height_range[0], height_range[1] + 1)
            width = np.random.randint(width_range[0], width_range[1] + 1)
            figure_type = Image.get_figure_type(r % num_classes)
            base = DataGenerator.create_base(figure_type, height, width)
            base.add_noise(noise)
            if np.random.uniform() < 0.7 and figure_type != "vertical bars":
                base.rotate_90()
            base.add_background(n, center)
            dataset.add(base)

        return dataset

    @staticmethod
    def create_base(base_type, height, width):
        """
        Creates a base image object without the background.
        :param base_type: Figure type, i.e. 'circle', 'rectangle' etc.
        :param height: Height of the figure.
        :param width: Width of the figure.
        :return: Image object.
        """
        base = np.zeros((height, width))

        if base_type == "vertical bars":
            num_bars = int(height / 5)
            num_bars = 1 if num_bars == 0 else num_bars
            for k in range(num_bars):
                i = np.random.randint(0, height)
                base[i, :] = 1

        elif base_type == "circle":
            height = min(height, width)
            width = height
            base = np.zeros((height, width))
            r1 = (height - 1) / 2
            r2 = (width - 1) / 2
            step = 1 if height >= 20 else 5
            for angle in range(0, 360, step):
                i = r1 * np.cos(np.radians(angle)) + np.floor(height / 2)
                j = r2 * np.sin(np.radians(angle)) + np.floor(width / 2)
                base[int(i), int(j)] = 1

        elif base_type == "rectangle":
            base[:, [0, -1]] = 1
            base[[0, -1], :] = 1

        elif base_type == "cross":
            h = int(np.ceil(height / 2 - 1))
            w = int(np.ceil(width / 2 - 1))

            for i in range(height):
                if i == h:
                    for j in range(width):
                        base[i, j] = 1
                else:
                    base[i, w] = 1
        base = Image(base, figure=base_type)
        return base


class DataSet:
    """
    Class for data sets.
    """

    def __init__(self):
        """
        Initializes variables to default values.
        """

        self.data_dict = {}
        self.size = 0

    def add(self, image):
        """
        Adds an image to the data set.
        :param image: Image object.
        :return: None
        """
        image_class = image.get_label()
        if not self.data_dict.get(image_class, False):
            self.data_dict[image_class] = []
            self.data_dict[image_class].append(image)
        else:
            self.data_dict[image_class].append(image)

        self.size += 1

    def add_multiple(self, images):
        """
        Adds multiple images to the dataset.
        :param images: List of Image objects.
        :return: None
        """
        for image in images:
            self.add(image)

    def flatten(self, one_hot=True):
        """
        Flattens the DataSet object to numpy arrays.
        :param one_hot: True if targets should be one hot.
        :return: Two numpy arrays, one for the flattened images and one for the targets/labels.
        """

        flat_data = []
        labels = []

        for c in self.data_dict:
            class_list = self.data_dict[c]
            for image in class_list:
                flat_data.append(image.to_1d())
                if one_hot:
                    one_hot_vector = [0 for _ in range(len(Image.class_dict))]
                    one_hot_vector[image.get_label()] = 1
                    labels.append(one_hot_vector)
                else:
                    labels.append([image.get_label()])

        return np.array(flat_data), np.array(labels)

    @staticmethod
    def _flatten(data_list, num_classes=4, one_hot=True):
        """
        Static method for flattening a list of images to numpy arrays.
        :param data_list: List of Image objects.
        :param num_classes: Number of classes.
        :param one_hot: True if targets should be one hot.
        :return: Two numpy arrays, one for the flattened images and one for the targets/labels.
        """

        flat_data = []
        labels = []

        for image in data_list:
            flat_data.append(image.to_1d())
            if one_hot:
                one_hot_vector = [0 for _ in range(num_classes)]
                one_hot_vector[image.get_label()] = 1
                labels.append(one_hot_vector)
            else:
                labels.append([image.get_label()])

        return np.array(flat_data), np.array(labels)

    def split(self, train_size=0.7, test_size=0.1, val_size=0.2):
        """
        Splits the DataSet object into training, testing and validation sets.
        :param train_size: Size of training set.
        :param test_size: Size of test set.
        :param val_size:  Size of validation set,
        :return: Three DataSet objects corresponding to training, testing and validation.
        """
        num_classes = len(self.data_dict.keys())

        training = DataSet()
        testing = DataSet()
        validation = DataSet()
        for i in range(num_classes):
            num_each_train = int(np.ceil(len(self.data_dict[i]) * train_size))
            num_each_val = int(np.ceil(len(self.data_dict[i]) * val_size))

            training.add_multiple(self.data_dict[i][:num_each_train])
            validation.add_multiple(self.data_dict[i][num_each_train:num_each_train + num_each_val])
            testing.add_multiple(self.data_dict[i][num_each_train + num_each_val:])

        return training, testing, validation

    def visualize(self, num=10, ncols=5):
        """
        Method for visualizing the DataSet object.
        :param num: Number of images to display.
        :param ncols: Number of images per row.
        :return: None
        """
        nrows = int(np.ceil(num / ncols))
        fig = plt.figure()
        num_classes = len(self.data_dict.keys())
        for i in range(num):
            c = i % num_classes
            image = np.random.choice(self.data_dict[c])

            a = fig.add_subplot(nrows, ncols, i + 1)
            a.axis('off')
            imgplot = plt.imshow(image.flat)
            a.set_title(image.get_figure())
        fig.show()


class Image:
    """
    Class for images.
    """
    class_dict = {
        "vertical bars": 0,
        "circle": 1,
        "rectangle": 2,
        "cross": 3,
    }

    def __init__(self, flat, figure):
        """
        Initializes variables.
        :param flat: 2D Numpy representation of the image.
        :param figure: String indicating figure type.
        """
        self.figure = figure
        self.label = Image.class_dict[figure]
        self.flat = np.array(flat)

    @staticmethod
    def get_figure_type(c):
        """
        :param c: Integer indicating class.
        :return: String representation of the class.
        """
        return list(Image.class_dict.keys())[list(Image.class_dict.values()).index(c)]

    def to_1d(self):
        """
        :return: 1D Numpy array representation of the image.
        """
        return np.reshape(self.flat, (-1,))

    def get_label(self):
        """
        :return: Integer indicating class.
        """
        return self.label

    def get_figure(self):
        """
        :return: String indicating class.
        """
        return self.figure

    def add_noise(self, noise):
        """
        Method that adds noise to the image given noise parameter.
        :param noise: Float between 0 and 1.
        :return: None
        """
        size = self.flat.size
        noise = int(np.ceil(size * noise))
        for _ in range(noise):
            i = np.random.randint(0, self.flat.shape[0])
            j = np.random.randint(0, self.flat.shape[1])
            self.flat[i, j] = 1 if self.flat[i, j] == 0 else 0

    def rotate_90(self):
        """Rotates the image by 90 degrees."""
        self.flat = np.rot90(self.flat)

    def add_background(self, n, center=True):
        """
        Adds a background to the image.
        :param n: Integer indicating size (nxn).
        :param center: True if the image/object should be centered on the background, False otherwise.
        :return: None
        """
        if n < self.flat.shape[0] or n < self.flat.shape[1]:
            raise ValueError("The size of the nxn background cannot be smaller than figure")
        choice = np.random.choice(['upper left', 'upper right', "down left", "down right", "center"])
        if center or choice == "center":
            rows = int((n - self.flat.shape[0]) / 2)
            rest_rows = int((n - self.flat.shape[0]) % 2)
            columns = int((n - self.flat.shape[1]) / 2)
            rest_columns = int((n - self.flat.shape[1]) % 2)
            self.flat = np.pad(self.flat, ((rows, rows + rest_rows), (columns, columns + rest_columns)), 'constant',
                               constant_values=0)
        else:
            rows = n - self.flat.shape[0]
            columns = n - self.flat.shape[1]
            if choice == 'upper left':
                self.flat = np.pad(self.flat, ((0, rows), (0, columns)), 'constant',
                                   constant_values=0)
            elif choice == 'upper right':
                self.flat = np.pad(self.flat, ((0, rows), (columns, 0)), 'constant',
                                   constant_values=0)
            elif choice == 'down left':
                self.flat = np.pad(self.flat, ((rows, 0), (0, columns)), 'constant',
                                   constant_values=0)
            elif choice == 'down right':
                self.flat = np.pad(self.flat, ((rows, 0), (columns, 0)), 'constant',
                                   constant_values=0)


if __name__ == "__main__":
    pass
