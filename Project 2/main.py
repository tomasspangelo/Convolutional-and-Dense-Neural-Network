import sys
from configparser import ConfigParser

import matplotlib.pyplot as plt
from data_generator import DataGenerator
from layers import Input, Conv2D, Conv1D, FullyConnected, Softmax
from network import Network


def add_channel(x):
    shape = x.shape[0:1] + (1,) + x.shape[1:]
    return x.reshape(shape)


def main_test():
    data_generator = DataGenerator()

    dataset = data_generator.generate_data(n=20,
                                           height_range=(10, 20),
                                           width_range=(10, 20),
                                           noise=0.01,
                                           size=143,
                                           num_classes=4,
                                           center=True)

    train_set, test_set, val_set = dataset.split(train_size=0.7,
                                                 test_size=0.1,
                                                 val_size=0.2,
                                                 )

    x_train, y_train = train_set.flatten_2D()
    x_val, y_val = val_set.flatten_2D()
    x_test, y_test = test_set.flatten_2D()

    x_train = add_channel(x_train)
    x_val = add_channel(x_val)
    x_test = add_channel(x_test)

    model = Network()

    model.add(Input(input_size=(20, 20)))

    model.add(Conv2D(activation="relu",
                     kernel_size=(4, 4),
                     num_kernels=4,
                     stride=(1, 1),
                     mode=("same", "same"),
                     ))
    model.add(Conv2D(activation="relu",
                     kernel_size=(4, 4),
                     num_kernels=2,
                     stride=(1, 1),
                     mode=("same", "same"),
                     ))
    model.add(FullyConnected(neurons=4,
                             activation="tanh"
                             ))

    model.add(Softmax())

    model.compile(loss="cross_entropy",
                  regularization="l2",
                  reg_rate=0.001,
                  learning_rate=0.1,
                  )

    train_loss, val_loss = model.fit(train_data=x_train,
                                     targets=y_train,
                                     batch_size=4,
                                     epochs=50,
                                     metrics=[],
                                     val_data=x_val,
                                     val_targets=y_val,
                                     verbosity=2)

    test_loss = model.evaluate(x_test, y_test)
    fig2 = plt.figure()
    plt.plot(train_loss, 'b')
    plt.plot(val_loss, 'g')
    plt.plot(11 - 1, test_loss, 'ro')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(('train_loss', 'val_loss', "test_loss"), loc='upper right')
    fig2.show()
    print("Test loss: {test_loss}".format(test_loss=test_loss))
    test_accuracy = model.accuracy(x_test, y_test)
    print("Test accuracy: {test_accuracy}".format(test_accuracy=test_accuracy))

    model.save("50_epoch", [train_loss, val_loss, test_loss])
    input("Press enter to exit ")


def load_network(filename):
    model, loss = Network.load(filename, loss=True)
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Conv1D):
            layer.visualize_kernels()

    train_loss = loss[0]
    val_loss = loss[1]
    test_loss = loss[2]

    fig2 = plt.figure()
    plt.plot(train_loss, 'b')
    plt.plot(val_loss, 'g')
    plt.plot(len(train_loss) - 1, test_loss, 'ro')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(('train_loss', 'val_loss', "test_loss"), loc='upper right')
    fig2.show()


def init_dataset(data_config):
    print("Generating the dataset...")
    n = int(data_config['n'])
    size = int(data_config['size'])
    dimensions = int(data_config['dimensions'])

    data_generator = DataGenerator()

    if dimensions == 1:
        dataset = data_generator.generate_data_1D(n=n,
                                                  size=size,
                                                  num_classes=4)
    else:
        height_range = (int(data_config['min_height']), int(data_config['max_height']))
        width_range = (int(data_config['min_width']), int(data_config['max_width']))
        noise = float(data_config['noise'])
        center = data_config.getboolean('center')

        dataset = data_generator.generate_data_2D(n=n,
                                                  height_range=height_range,
                                                  width_range=width_range,
                                                  noise=noise,
                                                  size=size,
                                                  num_classes=4,
                                                  center=center)

    train_size = float(data_config['train_size'])
    test_size = float(data_config['test_size'])
    val_size = float(data_config['val_size'])
    train_set, test_set, val_set = dataset.split(train_size=train_size,
                                                 test_size=test_size,
                                                 val_size=val_size,
                                                 )
    if train_set.size == 0 or test_set.size == 0 or val_set.size == 0:
        raise ValueError("Dataset is too small to split.")
    return dataset, train_set, test_set, val_set


def init_network(network_config):
    model = Network()

    input_size = eval(network_config['input_size'])
    layers = eval(network_config['layers'])
    weight_ranges = eval(network_config['weight_ranges'])
    softmax = network_config.getboolean('softmax')
    kernel_shape = eval(network_config['kernel_shape'])
    # syntax: (num_kernels, mode, stride)
    conv_settings = eval(network_config['conv_settings'])

    model.add(Input(input_size=input_size))

    conv = 0
    conv_in_first = False
    for i in range(len(layers)):
        layer_type = layers[i][0]
        activation = layers[i][1]
        weight_range = weight_ranges[i]

        if layer_type.lower() == "dense":
            neurons = layers[i][2]
            model.add(FullyConnected(neurons=neurons,
                                     activation=activation,
                                     weight_range=weight_range
                                     ))
        elif layer_type.lower() == "conv1d":
            shape = kernel_shape[conv]
            num_kernels = conv_settings[conv][0]
            mode = conv_settings[conv][1]
            stride = conv_settings[conv][2]
            layer = Conv1D(activation=activation,
                           kernel_size=shape,
                           num_kernels=num_kernels,
                           stride=stride,
                           mode=mode,
                           weight_range=weight_range)
            model.add(layer)
            layer.visualize_kernels()
            if conv == 0:
                conv_in_first = True

            conv += 1
        elif layer_type.lower() == "conv2d":
            shape = kernel_shape[conv]
            num_kernels = conv_settings[conv][0]
            mode = conv_settings[conv][1]
            stride = conv_settings[conv][2]
            layer = Conv2D(activation=activation,
                           kernel_size=shape,
                           num_kernels=num_kernels,
                           stride=stride,
                           mode=mode,
                           weight_range=weight_range)
            model.add(layer)

            layer.visualize_kernels()
            if conv == 0:
                conv_in_first = True
            conv += 1

    if softmax:
        model.add(Softmax())

    loss = network_config['loss']
    regularization = network_config['regularization']
    reg_rate = float(network_config['reg_rate'])
    learning_rate = float(network_config['learning_rate'])

    model.compile(loss=loss,
                  regularization=regularization,
                  reg_rate=reg_rate,
                  learning_rate=learning_rate,
                  )

    return model, conv_in_first


def main():
    if len(sys.argv) < 3:
        print("Please indicate if file should be loaded [load/fit] + filename w/out file extension, and try again.")
        return

    if sys.argv[1] == "load":
        load_network(sys.argv[2])
        input("Press enter to exit ")
        return

    config = ConfigParser()
    config.read("./config/" + sys.argv[2] + ".ini")
    data_config = config['data_generation']

    dataset, train_set, test_set, val_set = init_dataset(data_config)
    print("Successfully generated dataset")

    dimensions = int(data_config['dimensions'])
    num = 10 if dimensions == 2 else 20
    dataset.visualize(num)


    network_config = config['network']
    model, conv_in_first = init_network(network_config)
    plt.show()
    epochs = int(config['fit']['epochs'])
    metrics = eval(config['fit']['metrics'])
    batch_size = int(config['fit']['batch_size'])
    verbosity = int(config['fit']['verbosity'])
    save = config['fit'].getboolean('save')

    x_train, y_train = train_set.flatten_1D() if dimensions == 1 else train_set.flatten_2D()
    x_val, y_val = val_set.flatten_1D() if dimensions == 1 else val_set.flatten_2D()
    x_test, y_test = test_set.flatten_1D() if dimensions == 1 else test_set.flatten_2D()

    if conv_in_first:
        x_train = add_channel(x_train)
        x_val = add_channel(x_val)
        x_test = add_channel(x_test)

    print("Training set shape: {shape}".format(shape=x_train.shape))
    print("Test set shape: {shape}".format(shape=x_test.shape))
    print("Validation set shape: {shape}".format(shape=x_val.shape))

    train_loss, val_loss = model.fit(train_data=x_train,
                                     targets=y_train,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     metrics=metrics,
                                     val_data=x_val,
                                     val_targets=y_val,
                                     verbosity=verbosity)

    test_loss = model.evaluate(x_test, y_test)
    fig2 = plt.figure()
    plt.plot(train_loss, 'b')
    plt.plot(val_loss, 'g')
    plt.plot(epochs - 1, test_loss, 'ro')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(('train_loss', 'val_loss', "test_loss"), loc='upper right')
    fig2.show()
    print("Test loss: {test_loss}".format(test_loss=test_loss))
    if "accuracy" in metrics:
        test_accuracy = model.accuracy(x_test, y_test)
        print("Test accuracy: {test_accuracy}".format(test_accuracy=test_accuracy))

    if save:
        filename = config['fit']['filename']
        model.save(filename, [train_loss, val_loss, test_loss])

    input("Press enter to exit ")


if __name__ == "__main__":
    main()
