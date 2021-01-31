import numpy as np
import matplotlib.pyplot as plt


# TODO: Bias regularization


# Class for Layer objects
class Layer:

    def __init__(self, input_size, neurons, activation):

        self.activation = None
        self.act = None
        self.sum_in = None
        self.prev_layer = None
        self.weights = np.array([])
        self.biases = np.array([])
        self.weight_gradient = 0
        self.bias_gradient = 0

        self.neurons = neurons
        self.input_size = input_size

        if activation == 'sigmoid':
            self.activation = self._sigmoid

        if activation == 'tanh':
            self.activation = self._tanh

        if activation == 'relu':
            self.activation = self._relu

        if activation == 'linear':
            self.activation = self._linear

        if activation == 'softmax':
            self.activation = self._softmax

    def _sigmoid(self, x, derivative=False):

        if derivative:
            return self._sigmoid(x) * (1 - self._sigmoid(x))

        return 1 / (1 + np.exp(-x))

    def _tanh(self, x, derivative=False):

        if derivative:
            return 1 - np.tanh(x) ** 2

        return np.tanh(x)

    def _relu(self, x, derivative=False):

        if derivative:
            der = np.copy(x)
            der[x > 0] = 1
            der[x <= 0] = 0
            return der

        return np.maximum(0, x)

    def _linear(self, x, derivative=False):

        if derivative:
            return np.ones(x.shape)

        return x

    def _softmax(self, x, derivative=False):
        if derivative:
            if len(x.shape) > 1:
                square = np.array([np.tile(row, (len(row), 1)) for row in x])
                square_trans = np.array([np.tile(row, (len(row), 1)).T for row in x])
                diag = np.array([np.diag(row) for row in x])
                return diag + square * -square_trans
            return np.diag(x) + np.tile(x, (len(x), 1)) * -np.tile(x, (len(x), 1)).T

        if len(x.shape) > 1:
            return np.array([np.exp(row) / np.sum(np.exp(row)) for row in x])

        return np.exp(x) / np.sum(np.exp(x))


class Input(Layer):
    def __init__(self, input_size):
        super().__init__(input_size=input_size, neurons=input_size, activation=None)
        self.input_size = input_size

    def forward_pass(self, x):
        self.sum_in = x
        self.act = x

        return self.act

    def backward_pass(self, J_L_Z):
        raise TypeError("The input layer does not have a backward pass method.")


class FullyConnected(Layer):

    def __init__(self, neurons, activation, weight_range=(-0.1, 0.1)):
        super().__init__(input_size=0, neurons=neurons, activation=activation)
        if not self.activation:
            raise ValueError("'{activation}' is not a valid activation function for a fully connected layer.".format(
                activation=activation))
        self.weight_range = weight_range

    def forward_pass(self, x):
        self.sum_in = np.dot(x, self.weights) + self.biases
        self.act = self.activation(self.sum_in)

        return self.act

    def initialize_weights(self, input_size):
        # np.random.seed(3)
        self.weights = np.random.uniform(self.weight_range[0], self.weight_range[1], (input_size, self.neurons))

    def initialize_biases(self):
        # np.random.seed(5)
        self.biases = np.random.uniform(self.weight_range[0], self.weight_range[1], (self.neurons,))

    def backward_pass(self, J_L_Z):
        # TODO: double check if all matrix calculations are correct

        J_sum_diag = self.activation(self.sum_in, derivative=True)  # OK, diag(delta)
        J_Z_Y = np.einsum("ij,jk->ijk", J_sum_diag, np.transpose(self.weights))
        J_Z_W = np.einsum('ij,ik->ijk', self.prev_layer.act, J_sum_diag)  # Should be correct now

        # J_L_W = J_L_Z * J_Z_W  # Use this to update W / MUST BE FIXED
        # J_L_Y = np.dot(J_L_Z, J_Z_Y)  # Send this upstream / CHECK THIS

        J_L_W = np.einsum("ij,ikj->ikj", J_L_Z, J_Z_W)  # Should be correct now
        J_L_Y = np.einsum("ij,ijk->ik", J_L_Z, J_Z_Y)  # Should be correct now

        bias_gradient = J_L_Z * J_sum_diag

        self.weight_gradient = J_L_W
        self.bias_gradient = bias_gradient

        # I may have to return more than this to use downstream during backprop
        return J_L_Y


class Softmax(Layer):
    def __init__(self, neurons=0):
        super().__init__(input_size=neurons, neurons=neurons, activation='softmax')

    def forward_pass(self, x):
        self.sum_in = x
        self.act = self.activation(self.sum_in)

        return self.act

    def backward_pass(self, J_L_Z):
        J_soft = self.activation(self.act, derivative=True)

        return np.einsum("ij, ijk ->ik", J_L_Z, J_soft)


# Class for Network objects
class Network:

    def __init__(self):
        self.loss = None
        self.regularization = None
        self.layers = []
        self.reg_rate = 0

    def fit(self, train_data, targets, batch_size, epochs=50, val_data=None, val_targets=None):
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            training_loss = 0
            for i in range(0, train_data.shape[0], batch_size):
                mini_batch = train_data[i:i + batch_size]
                mini_batch_targets = targets[i:i + batch_size]  # TODO: Need to account for one-hot

                activation = self.forward_pass(mini_batch)

                # Should this be done AFTER parameters are updated or before?
                training_loss += self.loss(activation, mini_batch_targets)
                if self.regularization:
                    training_loss += sum(
                        [self.regularization(layer.weights) + self.regularization(layer.biases) for layer in
                         self.layers])

                self.backward_pass(activation, mini_batch_targets)
                self.update_parameters()

            training_loss = training_loss / train_data.shape[0]
            training_losses.append(training_loss)

            if val_data and val_targets:
                activation = self.forward_pass(val_data)

                validation_loss = 1 / val_data.shape[0] * self.loss(activation, val_targets)
                validation_losses.append(validation_loss)

        return training_losses, validation_losses

    def add(self, layer):
        if len(self.layers) > 0:
            if isinstance(layer, Input):
                raise ValueError("An input layer can only be the first layer")

            prev_layer = self.layers[-1]
            layer.prev_layer = prev_layer
            if not isinstance(layer, Softmax):
                layer.initialize_weights(input_size=prev_layer.neurons)
                layer.initialize_biases()
            else:
                layer.neurons = prev_layer.neurons
        else:
            if not isinstance(layer, Input):
                raise ValueError("The first layer in the network must be a input layer")

        self.layers.append(layer)

    def compile(self, loss, regularization, reg_rate, learning_rate):
        self.reg_rate = reg_rate
        num_layers = len(self.layers)
        for i in range(num_layers):
            layer = self.layers[i]
            layer.learning_rate = learning_rate
            if isinstance(layer, Softmax) and i != num_layers - 1:
                raise NotImplementedError("Network does not support Softmax layer other than at the output")

        if loss == "mse":
            self.loss = self._mse
        if loss == "cross_entropy":
            self.loss = self._cross_entropy

        if regularization == "l1":
            self.regularization = self._l1
        if regularization == "l2":
            self.regularization = self._l2

    def forward_pass(self, x):
        i = 1
        for layer in self.layers:
            x = layer.forward_pass(x)
            '''
            print("---------------")
            print("Layer {i}".format(i=i))
            print("Weights")
            print(layer.weights)
            print("Biases")
            print(layer.biases)
            print("Activations")
            print(x)
            print("---------------")
            '''
            i += 1

        return x

    def backward_pass(self, z, t):

        # Calculating the first of the Jacobians
        J = self.loss(z, t, derivative=True)

        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            J = layer.backward_pass(J)

    def update_parameters(self):
        for layer in self.layers:
            if not (isinstance(layer, Input) or isinstance(layer, Softmax)):

                # Calculate averaged weight gradient, add regularization term if applicable,
                # and update weights
                weight_gradient = layer.weight_gradient
                batch_w_gradient = 1 / weight_gradient.shape[0] * np.sum(weight_gradient, axis=0)
                if self.regularization:
                    batch_w_gradient += self.regularization(layer.weights, derivative=True)
                layer.weights = layer.weights - layer.learning_rate * batch_w_gradient

                # Calculate averaged bias gradient, add regularization term if applicable,
                # and update biases
                bias_gradient = layer.bias_gradient
                batch_b_gradient = 1 / bias_gradient.shape[0] * np.sum(bias_gradient, axis=0)
                if self.regularization:
                    batch_b_gradient += self.regularization(layer.biases, derivative=True)

                layer.biases = layer.biases - layer.learning_rate * batch_b_gradient

    def predict(self, x):

        # axis = 1 if x.ndim == 2 else 0
        # return np.argmax(self.forward_pass(x), axis=1)
        return self.forward_pass(x)

    def evaluate(self, x_test, y_test):
        activation = self.forward_pass(x_test)
        test_loss = self.loss(activation, y_test)
        if self.regularization:
            test_loss += sum(
                [self.regularization(layer.weights) + self.regularization(layer.biases) for layer in
                 self.layers])
        return test_loss

    def _mse(self, z, t, derivative=False):
        n = z.shape[-1]

        if derivative:
            return 2 / n * (z - t)

        axis = 1 if z.ndim == 2 else 0
        # return 1 / n * np.sum((z - t) ** 2, axis=axis)
        return sum(1 / n * np.sum((z - t) ** 2, axis=axis))

    def _cross_entropy(self, z, t, derivative=False):

        if derivative:
            return np.where(z != 0, -t * 1 / z, 0.0)
        axis = 1 if z.ndim == 2 else 0
        # return np.sum(t * np.log(z + 1.e-17), axis=axis)
        return sum(-np.sum(t * np.log(z + 1.e-17), axis=axis))

    def _l1(self, w, derivative=False):

        if derivative:
            return self.reg_rate * np.sign(w)
        return self.reg_rate * np.sum(np.absolute(w))

    def _l2(self, w, derivative=False):

        if derivative:
            return self.reg_rate * w

        return self.reg_rate * 1 / 2 * np.sum(w ** 2)


if __name__ == "__main__":
    model = Network()

    model.add(Input(input_size=2))
    model.add(FullyConnected(neurons=3,
                             activation='linear'
                             ))
    model.add(FullyConnected(neurons=2,
                             activation='sigmoid'
                             ))
    model.add(Softmax())

    a = np.array([1, 2, 3])
    b = np.array([3, 3, 3])
    c = np.array([[1, 2], [3, 3], [1, 1]])

    # model.forward_pass(a)
    # model.forward_pass(b)
    model.compile(loss="cross_entropy", regularization="l1", reg_rate=0.001, learning_rate=0.1)

    # model.forward_pass(c)

    target = np.array([[0, 1], [1, 0], [0, 1]])

    print("---------------")
    print("BACKWARD PASS")

    # model.backward_pass(c, target)

    # fit(self, train_data, targets, batch_size, epochs=50, val_data=None, val_targets=None)
    train_loss, val_loss = model.fit(train_data=c, targets=target, batch_size=1, epochs=1000)

    print(train_loss[-1])
    print(val_loss)
    plt.plot(train_loss)
    plt.show()

    print(model.predict(c))
