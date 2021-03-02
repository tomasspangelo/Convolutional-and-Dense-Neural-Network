import numpy as np
from layers import Input, Softmax, Conv2D, Conv1D, FullyConnected


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
            if isinstance(layer, Conv2D):
                if isinstance(prev_layer, Conv2D):
                    layer.initialize_kernels(prev_layer.channels, input_size=prev_layer.get_output_size(flatten=False))
                else:
                    layer.initialize_kernels(1, input_size=prev_layer.input_size)
            elif isinstance(layer, Conv1D):
                if isinstance(prev_layer, Conv2D) or isinstance(prev_layer, Conv1D):
                    layer.add_input_channels(prev_layer.channels, input_size=prev_layer.get_output_size())
                else:
                    layer.initialize_kernels(1, input_size=prev_layer.input_size)
            elif isinstance(layer, FullyConnected):
                if isinstance(prev_layer, Conv2D):
                    input_size = prev_layer.channels * prev_layer.get_output_size()
                    layer.initialize_weights(input_size=input_size)
                elif isinstance(prev_layer, Conv1D):
                    input_size = prev_layer.channels * prev_layer.get_output_size()
                    layer.initialize_weights(input_size=input_size)
                else:
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
            if isinstance(layer.prev_layer, Conv2D) and isinstance(layer, Conv1D):
                shape = x.shape[:-2] + (-1,)
                x = x.reshape(shape)
            elif isinstance(layer.prev_layer, Conv2D) and isinstance(layer, FullyConnected):
                # shape = (x.shape[0], np.prod(x.shape[1:]))
                shape = (x.shape[0], -1)
                x = x.reshape(shape)
            elif isinstance(layer.prev_layer, Conv1D) and isinstance(layer, FullyConnected):
                shape = (x.shape[0], -1)
                x = x.reshape(shape)
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
            downstream_layer = self.layers[i+1]
            if isinstance(layer, Conv2D) and isinstance(downstream_layer, FullyConnected):
                shape = layer.sum_in.shape
                J = J.reshape(shape)
            elif isinstance(layer, Conv1D) and isinstance(downstream_layer, FullyConnected):
                shape = layer.sum_in.shape
                J = J.reshape(shape)
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
