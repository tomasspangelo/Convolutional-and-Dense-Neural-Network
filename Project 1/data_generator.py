import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:

    @staticmethod
    def generate_data(n, height_range, width_range, noise, size, num_classes=4, center=True):

        dataset = DataSet()
        per_class = int(size / num_classes)
        rest_class = int(size % num_classes)
        for c in range(num_classes):
            for _ in range(per_class):
                height = np.random.randint(height_range[0], height_range[1] + 1)
                width = np.random.randint(width_range[0], width_range[1] + 1)
                base = DataGenerator.create_base(Image.get_figure_type(c), height, width)
                base.add_noise(noise)
                if np.random.uniform() > 0.7:
                    base.rotate_90()
                base.add_background(n, center)
                dataset.add(base)

        for r in range(rest_class):
            height = np.random.randint(height_range[0], height_range[1] + 1)
            width = np.random.randint(width_range[0], width_range[1] + 1)
            figure_type = Image.get_figure_type(r % num_classes)
            base = DataGenerator.create_base(figure_type, height, width)
            base.add_noise(noise)
            if np.random.uniform() > 0.7 and figure_type != "vertical bars":
                base.rotate_90()
            base.add_background(n, center)
            dataset.add(base)

        return dataset

    @staticmethod
    def create_base(base_type, height, width):
        base = np.zeros((height, width))

        if base_type == "vertical bars":
            num_bars = int(height / 5)
            num_bars = 1 if num_bars == 0 else num_bars
            for k in range(num_bars):
                i = np.random.randint(0, height)
                base[i, :] = 1

        if base_type == "circle":
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
    def __init__(self):
        self.data_dict = {}
        self.size = 0

    def add(self, image):
        image_class = image.get_label()
        if not self.data_dict.get(image_class, False):
            self.data_dict[image_class] = []
            self.data_dict[image_class].append(image)
        else:
            self.data_dict[image_class].append(image)

        self.size += 1

    def add_multiple(self, images):
        for image in images:
            self.add(image)

    def flatten(self, one_hot=False):

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
    def _flatten(data_list, num_classes=0, one_hot=True):

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

    def split(self, train_size=0.7, test_size=0.1, val_size=0.2, one_hot=True):
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


class Image:
    class_dict = {
        "vertical bars": 0,
        "circle": 1,
        "rectangle": 2,
        "cross": 3,
    }

    def __init__(self, flat, figure):
        self.figure = figure
        self.label = Image.class_dict[figure]
        self.flat = np.array(flat)

    @staticmethod
    def get_figure_type(c):
        return list(Image.class_dict.keys())[list(Image.class_dict.values()).index(c)]

    def to_1d(self, vector=True):
        if vector:
            return np.reshape(self.flat, (-1,))

        return self.flat

    def get_label(self):
        return self.label

    def get_figure(self):
        return self.figure

    def add_noise(self, noise):
        size = self.flat.size
        noise = int(np.ceil(size * noise))
        for _ in range(noise):
            i = np.random.randint(0, self.flat.shape[0])
            j = np.random.randint(0, self.flat.shape[1])
            self.flat[i, j] = 1 if self.flat[i, j] == 0 else 0

    def rotate_90(self):
        self.flat = np.rot90(self.flat)

    def add_background(self, n, center=True):
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
    '''
    for i in range(1,100,5):
        base = datagen.create_base("circle", height=i, width=i)
    
    b = datagen.create_base("circle", height=25, width=25)
    b.add_background(30, center=False)
    flat = b.flat
    dataset = DataGenerator.generate_data(n=20,
                                          height_range=(15, 20),
                                          width_range=(15, 20),
                                          noise=0.01,
                                          size=16,
                                          num_classes=4)
    
    training_set, training_labels, testing_set, testing_labels, validation_set, validation_labels = dataset.split()
    print(dataset)
    '''
    # image1 = DataGenerator.create_base("circle", height=100, width=100)
    # image2 = DataGenerator.create_base("cross", height=10, width=50)
    image2 = DataGenerator.create_base("circle", height=20, width=50)
    image2.add_background(30)
    image2.add_background(50)
    plt.imshow(image2.flat)
    plt.show()
