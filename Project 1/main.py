import sys
import matplotlib.pyplot as plt
from configparser import ConfigParser
from data_generator import DataGenerator
from network import Network, Input, FullyConnected, Softmax


def init_dataset(data_config):
    print("Generating the dataset...")
    n = int(data_config['n'])
    height_range = (int(data_config['min_height']), int(data_config['max_height']))
    width_range = (int(data_config['min_width']), int(data_config['max_width']))
    noise = float(data_config['noise'])
    size = int(data_config['size'])
    center = data_config.getboolean('center')

    data_generator = DataGenerator()

    dataset = data_generator.generate_data(n=n,
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

    input_size = int(network_config['input_size'])
    layers = eval(network_config['layers'])  # Possible bug
    weight_ranges = eval(network_config['weight_ranges'])
    softmax = network_config.getboolean('softmax')

    model.add(Input(input_size=input_size))
    for i in range(len(layers)):
        neurons = layers[i][0]
        activation = layers[i][1]
        weight_range = weight_ranges[i]

        model.add(FullyConnected(neurons=neurons,
                                 activation=activation,
                                 weight_range=weight_range
                                 ))
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

    return model


def main():
    if len(sys.argv) < 2:
        print("No configuration file provided, try again.")
        return

    config = ConfigParser()
    config.read("./config/" + sys.argv[1])

    data_config = config['data_generation']
    dataset, train_set, test_set, val_set = init_dataset(data_config)
    dataset.visualize(10)

    network_config = config['network']
    model = init_network(network_config)

    epochs = int(config['fit']['epochs'])
    metrics = eval(config['fit']['metrics'])
    batch_size = int(config['fit']['batch_size'])
    verbosity = int(config['fit']['verbosity'])

    x_train, y_train = train_set.flatten()
    x_val, y_val = val_set.flatten()
    x_test, y_test = test_set.flatten()

    print("Successfully generated dataset")
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
    test_accuracy = model.accuracy(x_test, y_test)
    print("Test accuracy: {test_accuracy}".format(test_accuracy=test_accuracy))
    input("Press enter to exit ")


if __name__ == "__main__":
    main()
