import numpy as np


# Class for Layer objects


class Layer:

    def __init__(self, input_size, neurons, activation):

        self.activation = None
        self.act = None
        self.sum_in = None
        self.prev_layer = None
        self.weights = []
        self.biases = []

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
            return 1 - self.tanh(x) ** 2

        return np.tanh(x)

    def _relu(self, x, derivative=False):

        if derivative:
            return x > 0

        return np.maximum(0, x)

    def _linear(self, x, derivative=False):

        if derivative:
            return 1

        return x

    def _softmax(self, x):
        print("softmax input:")
        print(x)

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
        self.weight_range = weight_range

    def forward_pass(self, x):
        self.sum_in = np.dot(x, self.weights) + self.biases
        self.act = self.activation(self.sum_in)

        return self.act

    def initialize_weights(self, input_size):
        np.random.seed(3)
        self.weights = np.random.uniform(self.weight_range[0], self.weight_range[1], (input_size, self.neurons))

    def initialize_biases(self):
        np.random.seed(5)
        self.biases = np.random.uniform(self.weight_range[0], self.weight_range[1], (self.neurons,))

    def backward_pass(self, J_L_Z):
        # TODO: double check if all matrix calculations are correct
        # TODO: bias gradient

        J_sum_diag = self.activation(self.sum_in, derivative=True)  # OK, diag(delta)
        J_sum = np.array([np.diag(row) for row in J_sum_diag])  # OK, delta

        J_Z_Y = np.dot(J_sum, np.transpose(self.weights))  # OK (ISH)
        J_Z_W = np.outer(self.prev_layer.act, J_sum_diag)  # CHECK THIS

        J_L_W = np.dot(J_L_Z, J_Z_W)  # Use this to update W / CHECK THIS
        J_L_Y = np.dot(J_L_Z, J_Z_Y)  # Send this upstream / CHECK THIS

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
        raise NotImplementedError("The backward pass is not implemented for softmax.")


# Class for Network objects
class Network:

    def __init__(self):
        self.loss = None
        self.regularization = None
        self.layers = []

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

    def compile(self, loss, regularization):
        num_layers = len(self.layers)
        for i in range(num_layers-1):
            if isinstance(self.layers[i], Softmax):
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
            print("---------------")
            print("Layer {i}".format(i=i))
            print("Weights")
            print(layer.weights)
            print("Biases")
            print(layer.biases)
            print("Activations")
            print(x)
            print("---------------")
            i += 1

        return x

    def backward_pass(self, z, t):

        # Calculating the first of the Jacobians
        if self.loss == self._mse:
            j_loss = self._mse(z, t)

        if self.loss == self._cross_entropy:
            pass

        for i in range(len(self.layers) - 2, -1, -1):
            something = self.layers.backward_pass(1)

        return

    def _mse(self, z, t, derivative=False):
        n = z.size

        if derivative:
            return 2 / n * (z - t)

        return 1 / n * np.sum((z - t) ** 2)

    def _cross_entropy(self, y, t, derivative=False):

        if derivative:
            return

        return

    def _l1(self):
        return

    def _l2(self):
        return


if __name__ == "__main__":
    model = Network()

    model.add(Input(input_size=3))
    model.add(FullyConnected(neurons=2,
                             activation='relu'
                             ))
    model.add(Softmax())

    a = np.array([1, 2, 3])
    b = np.array([3, 3, 3])
    c = np.array([[1, 2, 3], [3, 3, 3], [1, 1, 1], [1, 2, 3]])

    # model.forward_pass(a)
    # model.forward_pass(b)
    model.forward_pass(a)

    layer1 = model.layers[0]

    #print(model.layers[0].sum_in)
    #print(layer1.activation(layer1.sum_in, derivative=True))
