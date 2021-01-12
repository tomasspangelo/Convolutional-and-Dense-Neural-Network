

# Class for Layer objects
# TODO: initial weight ranges
class Layer:

    def __init__(self, neurons, activation, weight_ranges):

        self.activation = None

        if activation == 'sigmoid':
            self.activation = self._sigmoid

        if activation == 'tanh':
            self.activation = self._tanh

        if activation == 'reulu':
            self.activation = self._reulu

        if activation == 'linear':
            self.activation = self._linear

        if activation == 'softmax':
            self.activation = self._softmax

        self.neurons = neurons
        self.weight_ranges = weight_ranges

    def forward_pass(self):
        return

    def backward_pass(self):
        return

    def _sigmoid(self):
        return

    def _tanh(self):
        return

    def _reulu(self):
        return

    def _linear(self):
        return

    def _softmax(self):
        return




# Class for Network objects
class Network:

    def __init__(self, layers, loss, regularization, reg_rate):
        self.layers = layers
        self.loss = loss  # TODO: Need to make internal functions for this (i.e. _mse)
        self.regularization = regularization  # TODO: Need to incorporate this in loss function

    def forward_pass(self):
        return

    def backward_pass(self):
        return

    def _mse(self):
        return

    def _cross_entropy(self):
        return

    def _l_one(self):
        return

    def _l_two(self):
        return



