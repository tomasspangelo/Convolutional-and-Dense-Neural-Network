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


class Network:
    """
    Class for (sequential) Network objects.
    """

    def __init__(self):
        """
        Initializes variables to default values.
        """
        self.loss = None
        self.regularization = None
        self.layers = []
        self.reg_rate = 0

    def fit(self, train_data, targets, batch_size, epochs=50, val_data=None, val_targets=None, metrics=[], verbosity=1):
        """
        The method that takes care of training the network (forward pass, backward pass and updates)
        :param train_data: Training data. Shape is (#samples, #features)
        :param targets: Targets for training data. Shape is (#samples, -1)
        :param batch_size: Number of samples in each mini batch.
        :param epochs: Number of epochs.
        :param val_data: Validation data. Shape is (#samples, #features)
        :param val_targets: Targets for validation data. Shape is (#samples, -1)
        :param metrics: List of additional metrics. Currently only supports accuracy.
        :param verbosity: 1 if user wants command line prints during training, 0 otherwise. 2 for more detailed info.
        :return: Training and validation losses for each epoch.
        """
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            if verbosity >= 1:
                print("Epoch {epoch}/{epochs}".format(epoch=epoch + 1, epochs=epochs))
            training_loss = 0
            for i in range(0, train_data.shape[0], batch_size):
                mini_batch = train_data[i:i + batch_size]
                mini_batch_targets = targets[i:i + batch_size]

                activation = self.forward_pass(mini_batch)
                batch_loss = self.loss(activation, mini_batch_targets)
                training_loss += batch_loss

                self.backward_pass(activation, mini_batch_targets)
                self.update_parameters()

                if verbosity == 2:
                    print("Mini batch input:")
                    print(mini_batch)
                    print("Mini batch targets:")
                    print(mini_batch_targets)
                    print("Network outpus:")
                    print(activation)
                    print("Mini batch loss: {batch_loss}".format(batch_loss=batch_loss))

            training_loss *= 1 / train_data.shape[0]
            '''
            if self.regularization:
                training_loss += sum(
                    [self.regularization(layer.weights) + self.regularization(layer.biases) for layer in
                     self.layers])
            '''
            training_losses.append(training_loss)

            if verbosity >= 1:
                print("Training loss: {training_loss}".format(training_loss=training_loss))
            if 'accuracy' in metrics and verbosity >= 1:
                training_accuracy = self.accuracy(train_data, targets)
                print("Training accuracy: {training_accuracy}".format(training_accuracy=training_accuracy))

            if val_data is not None and val_targets is not None:
                activation = self.forward_pass(val_data)

                validation_loss = self.loss(activation, val_targets)
                validation_loss *= 1 / val_data.shape[0]
                '''
                if self.regularization:
                    validation_loss += sum(
                        [self.regularization(layer.weights) + self.regularization(layer.biases) for layer in
                         self.layers])
                '''
                validation_losses.append(validation_loss)
                if verbosity >= 1:
                    print("Validation loss: {validation_loss}".format(validation_loss=validation_loss))
                if 'accuracy' in metrics and verbosity >= 1:
                    validation_accuracy = self.accuracy(val_data, val_targets)
                    print("Validation accuracy: {validation_accuracy}".format(validation_accuracy=validation_accuracy))

        return training_losses, validation_losses

    def add(self, layer):
        """
        Method for adding layers to the network.
        :param layer: Layer object.
        :return: None
        """
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
        """
        Method for initializing parameters important for training.
        Call this method after all layers have been added.
        :param loss: String indicating loss function.
        :param regularization: String indicating type of regularization.
        :param reg_rate: Regularization rate.
        :param learning_rate: Learning rate.
        :return: None
        """
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
        """
        Forwards pass for the network.
        :param x: Input data.
        :return: Activation of last layer.
        """
        for layer in self.layers:
            x = layer.forward_pass(x)

        return x

    def backward_pass(self, z, t):
        """
        Backward pass for the network.
        :param z: Activation of last layer.
        :param t: Targets for the input.
        :return: None
        """

        # Calculating the first of the Jacobians
        J = self.loss(z, t, derivative=True)

        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            J = layer.backward_pass(J)

    def update_parameters(self):
        """
        Updates the weights and biases for each (applicable) layer in the network.
        :return: None
        """
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

    def accuracy(self, x, t):
        """
        Calculates accuracy.
        :param x: Input data.
        :param t: Targets.
        :return: Accuracy
        """
        prediction = np.argmax(self.predict(x), axis=1)
        real_class = np.argmax(t, axis=1)
        accuracy = 0
        num_samples = len(prediction)
        for i in range(num_samples):
            if prediction[i] == real_class[i]:
                accuracy += 1
        return accuracy / num_samples

    def predict(self, x):
        """
        :param x: Input data.
        :return: Output of network.
        """

        return self.forward_pass(x)

    def evaluate(self, x_test, y_test):
        """
        Calculates loss.
        :param x_test: Input data.
        :param y_test: Input targets.
        :return: Averaged loss.
        """
        activation = self.forward_pass(x_test)
        test_loss = self.loss(activation, y_test)
        test_loss *= 1 / len(y_test)
        '''
        if self.regularization:
            test_loss += sum(
                [self.regularization(layer.weights) + self.regularization(layer.biases) for layer in
                 self.layers])
        '''
        return test_loss

    def _mse(self, z, t, derivative=False):
        """
        Mean Squared Error loss function.
        :param z: Output of network.
        :param t: Targets.
        :param derivative: True if method should return the derivative, False otherwise.
        :return: Loss (or derivative of loss). Not averaged.
        """
        n = z.shape[-1]
        if derivative:
            return 2 / n * (z - t)

        axis = 1 if z.ndim == 2 else 0
        return sum(1 / n * np.sum((z - t) ** 2, axis=axis))

    def _cross_entropy(self, z, t, derivative=False):
        """
        Categorical Cross Entropy loss function.
        :param z: Output of network.
        :param t: Targets.
        :param derivative: True if method should return the derivative, False otherwise.
        :return: Loss (or derivative of loss). Not averaged.
        """
        if derivative:
            return np.where(z != 0, -t * 1 / z, 0.0)
        axis = 1 if z.ndim == 2 else 0
        return sum(-np.sum(t * np.log(z), axis=axis))

    def _l1(self, w, derivative=False):
        """
        L1 regularization.
        :param w: Numpy array containing weights or biases.
        :param derivative: True if method should return the derivative, False otherwise.
        :return: Regularization. Note: multiplied by regularization rate.
        """

        if derivative:
            return self.reg_rate * np.sign(w)
        return self.reg_rate * np.sum(np.absolute(w))

    def _l2(self, w, derivative=False):
        """
        L2 regularization.
        :param w: Numpy array containing weights or biases.
        :param derivative: True if method should return the derivative, False otherwise.
        :return: Regularization. Note: multiplied by regularization rate.
        """
        if derivative:
            return self.reg_rate * w

        return self.reg_rate * 1 / 2 * np.sum(w ** 2)


if __name__ == "__main__":
    pass
