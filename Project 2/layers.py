import numpy as np


class Layer:
    """
    Superclass that all Layer classes inherits from.
    """

    def __init__(self, input_size, neurons, activation):
        """
        Initializes all variables and finds correct activation function
        according to the string input.
        :param input_size: Size of input to the layer.
        :param neurons: Number of neurons in the layer.
        :param activation: String indicating activation function.
        """

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

        elif activation == 'tanh':
            self.activation = self._tanh

        elif activation == 'relu':
            self.activation = self._relu

        elif activation == 'linear':
            self.activation = self._linear

        elif activation == 'softmax':
            self.activation = self._softmax

    def _sigmoid(self, x, derivative=False):
        """
        Sigmoid activation function.
        :param x: Input.
        :param derivative: True if method should return the derivative, otherwise False.
        :return: Activation or derivative depending on derivative param.
        """

        if derivative:
            return self._sigmoid(x) * (1 - self._sigmoid(x))

        return 1 / (1 + np.exp(-x))

    def _tanh(self, x, derivative=False):
        """
        Tanh activation function.
        :param x: Input.
        :param derivative: True if method should return the derivative, otherwise False.
        :return: Activation or derivative depending on derivative param.
        """
        if derivative:
            return 1 - np.tanh(x) ** 2

        return np.tanh(x)

    def _relu(self, x, derivative=False):
        """
        ReLu activation function.
        :param x: Input.
        :param derivative: True if method should return the derivative, otherwise False.
        :return: Activation or derivative depending on derivative param.
        """

        if derivative:
            der = np.copy(x)
            der[x > 0] = 1
            der[x <= 0] = 0
            return der

        return np.maximum(0, x)

    def _linear(self, x, derivative=False):
        """
        Linear activation function.
        :param x: Input.
        :param derivative: True if method should return the derivative, otherwise False.
        :return: Activation or derivative depending on derivative param.
        """

        if derivative:
            return np.ones(x.shape)

        return x

    def _softmax(self, x, derivative=False):
        """
        Linear activation function.
        :param x: Input.
        :param derivative: True if method should return the derivative, otherwise False.
        :return: Activation or derivative depending on derivative param.
        """
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
    """
    Class for Input layer, inherits from superclass Layer.
    """

    def __init__(self, input_size):
        """
        Initializes variables and calls constructor of superclass.
        :param input_size: Number of features for each data sample.
        """
        super().__init__(input_size=input_size, neurons=input_size, activation=None)
        self.input_size = input_size

    def forward_pass(self, x):
        """
        Forward pass for input layer, caches sum in and activation.
        :param x: Data, with one row per sample in minibatch. Shape is (#samples,#features)
        :return: Activation, which is the input.
        """
        self.sum_in = x
        self.act = x

        return self.act

    def backward_pass(self, J_L_Z):
        raise TypeError("The input layer does not have a backward pass method.")


class FullyConnected(Layer):
    """
    Class for Fully Connected layers, inherits from superclass Layer.
    """

    def __init__(self, neurons, activation, weight_range=(-0.5, 0.5)):
        """
        Initializes variables and calls superclass constructor.
        :param neurons: Number of neurons in the layer.
        :param activation: String indicating activation function.
        :param weight_range: Tuple indicating max and min weight value (for initialization).
        """
        super().__init__(input_size=0, neurons=neurons, activation=activation)
        if not self.activation:
            raise ValueError("'{activation}' is not a valid activation function for a fully connected layer.".format(
                activation=activation))
        self.weight_range = weight_range

    def forward_pass(self, x):
        """
        Forward pass for fully connected layer, caches sum in and activation.
        :param x: Input (one row in matrix per sample in minibatch)
        :return: Activation for layer.
        """
        self.sum_in = np.dot(x, self.weights) + self.biases
        self.act = self.activation(self.sum_in)

        return self.act

    def initialize_weights(self, input_size):
        """
        Initializes weights randomly by uniform distribution according to weight range.
        :param input_size: Size of the input (typically # neurons in previous layer).
        :return: None
        """
        self.weights = np.random.uniform(self.weight_range[0], self.weight_range[1], (input_size, self.neurons))

    def initialize_biases(self):
        """
        Initializes biases randomly by uniform distribution according to weight range.
        :return: None
        """
        self.biases = np.random.uniform(self.weight_range[0], self.weight_range[1], (self.neurons,))

    def backward_pass(self, J_L_Z):
        """
        Backward pass for a fully connected layer, cache gradients for update.
        :param J_L_Z: Derivative of the loss w.r.t. the output of this layer.
        :return: J_L_Y: Derivative of the loss w.r.t. the output of the previous layer.
        """

        # Derivative of output w.r.t. the sum in (diagonal of Jacobian)
        J_sum_diag = self.activation(self.sum_in, derivative=True)

        # Derivative of output w.r.t the output of the previous layer.
        # Corresponds to J_sum_diag dot W.T
        J_Z_Y = np.einsum("ij,jk->ijk", J_sum_diag, np.transpose(self.weights))

        # Derivative of output w.r.t the output of this layer.
        # Corresponds to Y outer product J_sum_diag
        J_Z_W = np.einsum('ij,ik->ijk', self.prev_layer.act, J_sum_diag)

        # Derivative of loss w.r.t the weights of this layer.
        # Every jth element of J_L_Z is multiplied by every item
        # in the jth column of J_Z_W
        J_L_W = np.einsum("ij,ikj->ikj", J_L_Z, J_Z_W)

        # Derivative of loss w.r.t the output of the previous layer.
        # Corresponds to J_L_Z dot J_Z_Y
        J_L_Y = np.einsum("ij,ijk->ik", J_L_Z, J_Z_Y)

        # Derivative of loss w.r.t the biases of this layer.
        # element-wise multiplication of J_L_Z and J_sum_diag
        bias_gradient = J_L_Z * J_sum_diag

        self.weight_gradient = J_L_W
        self.bias_gradient = bias_gradient

        return J_L_Y


class Softmax(Layer):
    """
    Class for Softmax layers. Inherits from superclass Layer.
    """

    def __init__(self, neurons=0):
        """
        Calls constructor of superclass.
        :param neurons: Number of neurons (should be same as # neurons in previous layer)
        """
        super().__init__(input_size=neurons, neurons=neurons, activation='softmax')

    def forward_pass(self, x):
        """
        Forward pass for softmax layer.
        :param x: Input (one row in matrix per sample in minibatch)
        :return: Activation for layer.
        """
        self.sum_in = x
        self.act = self.activation(self.sum_in)

        return self.act

    def backward_pass(self, J_L_Z):
        """
        Backward pass for softmax layer.
        :param J_L_Z: Derivative of loss w.r.t. output of this layer.
        :return: Derivative of loss w.r.t the output of the previous layer.
        """

        # Derivative of softmax output w.r.t output of the previous layer.
        J_soft = self.activation(self.act, derivative=True)

        # Derivative of loss w.r.t to output of prev. layer.
        # Corresponds to J_L_Z dot J_soft
        J_L_Y = np.einsum("ij, ijk ->ik", J_L_Z, J_soft)
        return J_L_Y
