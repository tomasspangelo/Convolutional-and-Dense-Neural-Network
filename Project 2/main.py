from data_generator import DataGenerator
from layers import Input
from network import Network


def add_channel(x):
    shape = x.shape[0:1] + (1,) + x.shape[1:]
    return x.reshape(shape)


def main():
    data_generator = DataGenerator()

    dataset = data_generator.generate_data(n=20,
                                           height_range=(10, 20),
                                           width_range=(10, 20),
                                           noise=0.01,
                                           size=200,
                                           num_classes=4,
                                           center=True)


    train_set, test_set, val_set = dataset.split(train_size=0.7,
                                                 test_size=0.1,
                                                 val_size=0.2,
                                                 )

    x_train, y_train = train_set.flatten()
    x_val, y_val = val_set.flatten()
    x_test, y_test = test_set.flatten()

    x_train = add_channel(x_train)
    x_val = add_channel(x_val)
    x_test = add_channel(x_test)

    model = Network()

    model.add(Input(input_size=input_size))








if __name__ == "__main__":
    main()
