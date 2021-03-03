from data_generator import DataGenerator
from layers import Input, Conv2D, FullyConnected, Softmax
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

    x_train, y_train = train_set.flatten_2D()
    x_val, y_val = val_set.flatten_2D()
    x_test, y_test = test_set.flatten_2D()

    x_train = add_channel(x_train)
    x_val = add_channel(x_val)
    x_test = add_channel(x_test)

    model = Network()

    model.add(Input(input_size=(20, 20)))

    model.add(Conv2D(activation="relu",
                     kernel_size=(4,4),
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
                                     epochs=20,
                                     metrics=['accuracy'],
                                     val_data=x_val,
                                     val_targets=y_val,
                                     verbosity=1)








if __name__ == "__main__":
    main()
