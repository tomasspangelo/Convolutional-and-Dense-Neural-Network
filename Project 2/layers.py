from collections import defaultdict

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


class Conv1D(Layer):
    def __init__(self, activation, kernel_size, num_kernels, stride, mode, weight_range=(-0.5, 0.5)):
        # TODO: Fix input size and neurons for Conv1D (and Conv2D)
        super().__init__(input_size=0, neurons=0, activation=activation)
        if not self.activation:
            raise ValueError("'{activation}' is not a valid activation function for a fully connected layer.".format(
                activation=activation))
        self.kernels = []
        self.channels = num_kernels
        self.channels_in = 1  # Remove?
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.stride = stride
        self.mode = mode
        self.weight_range = weight_range

    def forward_pass(self, x):

        batch_size = x.shape[0]
        left_padding, right_padding, output_width = self._add_padding(x[-1, -1], pad=False)
        padded_width = len(x[-1, -1]) + left_padding + right_padding
        sum_in = np.zeros((batch_size, self.channels, output_width))

        weight_dict = defaultdict(lambda: [])
        input_dict = defaultdict(lambda: [])

        for sample_num in range(len(x)):
            sample = x[sample_num]
            for i in range(self.channels):
                index = 0
                for j in range(0, padded_width - self.kernel_size + 1, self.stride):
                    for k in range(self.channels_in):
                        sample_in_channel, _ = self._add_padding(sample[k])
                        for l in range(self.kernel_size):
                            original_j = j + l - left_padding
                            sum_in[sample_num, i, index] += sample_in_channel[j + l] * self.kernels[i, k, l]
                            weight_dict[(i, k, l)].append((sample_in_channel[j + l], (sample_num, i, index)))
                            if 0 <= original_j < padded_width - left_padding - right_padding:
                                input_dict[(sample_num, k, original_j)].append(
                                    (self.kernels[i, k, l], (sample_num, i, index)))
                    index += 1
        self.sum_in = sum_in
        self.weight_dict = weight_dict
        self.input_dict = input_dict
        return self.activation(sum_in)

    def _add_padding(self, x, pad=True):
        """
        Adds padding to a single data point.
        :param x: sample x.
        :return: x with padding
        """
        input_width = len(x)
        if self.mode == "valid":
            output_width = int((input_width - self.kernel_size + 1) / self.stride)
            return x, output_width if pad else 0, 0, output_width

        if self.mode == "same":
            output_width = int(np.ceil(input_width / self.stride))
        elif self.mode == "full":
            output_width = int(np.ceil((input_width + self.kernel_size - 1) / self.stride))
        total_padding = (output_width - 1) * self.stride + self.kernel_size - input_width
        left_padding = int(total_padding / 2)
        right_padding = total_padding - left_padding
        if not pad:
            return left_padding, right_padding, output_width
        return np.pad(x, (left_padding, right_padding), 'constant', constant_values=0), output_width

    def get_output_size(self):
        input_width = self.input_size
        if self.mode == "valid":
            output_width = int((input_width - self.kernel_size + 1) / self.stride)
            return output_width

        if self.mode == "same":
            output_width = int(np.ceil(input_width / self.stride))
        elif self.mode == "full":
            output_width = int(np.ceil((input_width + self.kernel_size - 1) / self.stride))
        return output_width

    def initialize_kernels(self, channels_in, input_size):
        self.input_size = input_size
        self.channels_in = channels_in
        self.kernels = np.random.uniform(self.weight_range[0], self.weight_range[1],
                                         (self.num_kernels, channels_in, self.kernel_size))

    def backward_pass(self, J_L_Z):

        # (sample_num, i, index)
        # Derivative of output w.r.t. the sum in (diagonal of Jacobian)
        # J_sum_diag = self.activation(self.sum_in, derivative=True)

        # Derivative of output w.r.t the output of the previous layer.
        # Corresponds to J_sum_diag dot W.T
        # J_Z_Y = np.einsum("ij,jk->ijk", J_sum_diag, np.transpose(self.weights))

        J_L_K = np.zeros(self.kernels.shape)
        for key in self.weight_dict:
            for activation, sum_in in self.weight_dict[key]:
                # key[0]: channel i
                # key[1]: channel_in
                # key[2]: l, offset/index in kernel
                # sum[0]: sample_num
                # sum[1]: channel i
                # sum[2]: index
                num_sum_in = self.sum_in[sum_in[0], sum_in[1], sum_in[2]]
                delta = self.activation(num_sum_in, derivative=True) * J_L_Z[sum_in[0], sum_in[1], sum_in[2]]
                J_L_K[key[0], key[1], key[2]] += activation * delta

        batch_size = J_L_Z.shape[0]
        J_L_Y = np.zeros(self.prev_layer.sum_in.shape)  # TODO: Take reduction of dimensions into account
        if isinstance(self.prev_layer, Conv2D):
            J_L_Y = J_L_Y.reshape((batch_size, self.channels_in, -1))

        for key in self.input_dict:
            for kernel_weight, sum_in in self.input_dict[key]:
                # key[0]: sample_num
                # key[1]: channel_in
                # key[2]: original_j (index activation at Y)
                # sum[0]: sample_num
                # sum[1]: channel i
                # sum[2]: index
                num_sum_in = self.sum_in[sum_in[0], sum_in[1], sum_in[2]]
                delta = self.activation(num_sum_in, derivative=True) * J_L_Z[sum_in[0], sum_in[1], sum_in[2]]
                J_L_Y[key[0], key[1], key[2]] += kernel_weight * delta

        self.weight_gradient = J_L_K
        return J_L_Y if not isinstance(self.prev_layer, Conv2D) else J_L_Y.reshape(self.prev_layer.sum_in.shape)

    def update_parameters(self, regularization=None):
        weight_gradient = self.weight_gradient
        batch_w_gradient = 1 / weight_gradient.shape[0] * weight_gradient
        if regularization:
            batch_w_gradient += regularization(self.kernels, derivative=True)
        self.kernels = self.kernels - self.learning_rate * batch_w_gradient


class Conv2D(Layer):
    def __init__(self, activation, kernel_size, num_kernels, stride, mode, weight_range=(-0.5, 0.5)):
        super().__init__(input_size=0, neurons=0, activation=activation)
        self.kernels = []
        self.channels = num_kernels
        self.channels_in = 1  # Remove?
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.stride = stride
        self.mode = mode
        self.weight_range = weight_range

    def forward_pass(self, x):
        batch_size = x.shape[0]
        top_padding, bottom_padding, left_padding, right_padding, output_size = self._add_padding(x[-1, -1], pad=False)
        padded_height = x[-1, -1].shape[0] + top_padding + bottom_padding
        padded_width = x[-1, -1].shape[1] + left_padding + right_padding
        sum_in = np.zeros((batch_size, self.channels,) + output_size)

        weight_dict = defaultdict(lambda: [])
        input_dict = defaultdict(lambda: [])

        for sample_num in range(len(x)):
            sample = x[sample_num]
            for i in range(self.channels):
                r_index = 0
                for j in range(0, padded_height - self.kernel_size[0] + 1, self.stride[0]):
                    c_index = 0
                    for k in range(0, padded_width - self.kernel_size[1] + 1, self.stride[1]):
                        for l in range(self.channels_in):
                            sample_in_channel, _ = self._add_padding(sample[l])
                            for m in range(self.kernel_size[0]):
                                for n in range(self.kernel_size[1]):
                                    original_j = j + m - top_padding
                                    original_k = k + n - left_padding
                                    sum_in[sample_num, i, r_index, c_index] += sample_in_channel[j + m, k + n] * \
                                                                               self.kernels[i, l, m, n]
                                    weight_dict[(i, l, m, n)].append(
                                        (sample_in_channel[j + m, k + n], (sample_num, i, r_index, c_index)))
                                    original_height = padded_height - top_padding - bottom_padding
                                    original_width = padded_width - left_padding - right_padding
                                    if 0 <= original_j < original_height and 0 <= original_k < original_width:
                                        input_dict[(sample_num, l, original_j, original_k)].append(
                                            (self.kernels[i, l, m, n], (sample_num, i, r_index, c_index)))
                        c_index += 1
                    r_index += 1

        self.sum_in = sum_in
        self.weight_dict = weight_dict
        self.input_dict = input_dict
        self.act = self.activation(sum_in)
        return self.act

    def _add_padding(self, x, pad=True):
        """
        Adds padding to a single data point.
        :param x: sample x.
        :return: x with padding
        """

        input_height = x.shape[0]
        if self.mode[0] == "valid":
            output_height = int((input_height - self.kernel_size[0] + 1) / self.stride[0])
            top_padding = 0
            bottom_padding = 0
        else:
            if self.mode[0] == "same":
                output_height = int(np.ceil(input_height / self.stride[0]))
            elif self.mode[0] == "full":
                output_height = int(np.ceil((input_height + self.kernel_size[0] - 1) / self.stride[0]))
            total_padding = (output_height - 1) * self.stride[0] + self.kernel_size[0] - input_height
            top_padding = int(total_padding / 2)
            bottom_padding = total_padding - top_padding

        input_width = x.shape[1]
        if self.mode[1] == "valid":
            output_width = int((input_width - self.kernel_size[1] + 1) / self.stride[1])
            left_padding = 0
            right_padding = 0
        else:
            if self.mode[1] == "same":
                output_width = int(np.ceil(input_width / self.stride[1]))
            elif self.mode[1] == "full":
                output_width = int(np.ceil((input_width + self.kernel_size[1] - 1) / self.stride[1]))
            total_padding = (output_width - 1) * self.stride[1] + self.kernel_size[1] - input_width
            left_padding = int(total_padding / 2)
            right_padding = total_padding - left_padding
        if not pad:
            return top_padding, bottom_padding, left_padding, right_padding, (output_height, output_width)
        return np.pad(x, ((top_padding, bottom_padding), (left_padding, right_padding)), 'constant',
                      constant_values=0), (output_height, output_width)

    def get_output_size(self, flatten=True):
        input_height = self.input_size[0]
        if self.mode[0] == "valid":
            output_height = int((input_height - self.kernel_size[0] + 1) / self.stride[0])
        else:
            if self.mode[0] == "same":
                output_height = int(np.ceil(input_height / self.stride[0]))
            elif self.mode[0] == "full":
                output_height = int(np.ceil((input_height + self.kernel_size[0] - 1) / self.stride[0]))
        input_width = self.input_size[1]
        if self.mode[1] == "valid":
            output_width = int((input_width - self.kernel_size[1] + 1) / self.stride[1])
        else:
            if self.mode[1] == "same":
                output_width = int(np.ceil(input_width / self.stride[1]))
            elif self.mode[1] == "full":
                output_width = int(np.ceil((input_width + self.kernel_size[1] - 1) / self.stride[1]))
        return output_height * output_width if flatten else (output_height, output_width)

    def backward_pass(self, J_L_Z):

        # (sample_num, i, index)
        # Derivative of output w.r.t. the sum in (diagonal of Jacobian)
        # J_sum_diag = self.activation(self.sum_in, derivative=True)

        # Derivative of output w.r.t the output of the previous layer.
        # Corresponds to J_sum_diag dot W.T
        # J_Z_Y = np.einsum("ij,jk->ijk", J_sum_diag, np.transpose(self.weights))

        J_L_K = np.zeros(self.kernels.shape)
        for key in self.weight_dict:
            for activation, sum_in in self.weight_dict[key]:
                # key[0]: channel i
                # key[1]: channel_in
                # key[2]: m, offset/index in kernel
                # key[3]: n, offset/index in kernel
                # sum[0]: sample_num
                # sum[1]: channel i
                # sum[2]: index row
                # sum[3]: index column
                num_sum_in = self.sum_in[sum_in[0], sum_in[1], sum_in[2], sum_in[3]]
                delta = self.activation(num_sum_in, derivative=True) * J_L_Z[sum_in[0], sum_in[1], sum_in[2], sum_in[3]]
                J_L_K[key[0], key[1], key[2], key[3]] += activation * delta

        J_L_Y = np.zeros(self.prev_layer.sum_in.shape)

        for key in self.input_dict:
            for kernel_weight, sum_in in self.input_dict[key]:
                # key[0]: sample_num
                # key[1]: channel_in
                # key[2]: original_j (index activation at Y)
                # key[2]: original_k (index activation at Y)
                # sum[0]: sample_num
                # sum[1]: channel i
                # sum[2]: index
                num_sum_in = self.sum_in[sum_in[0], sum_in[1], sum_in[2], sum_in[3]]
                delta = self.activation(num_sum_in, derivative=True) * J_L_Z[sum_in[0], sum_in[1], sum_in[2], sum_in[3]]
                J_L_Y[key[0], key[1], key[2], key[3]] += kernel_weight * delta

        self.weight_gradient = J_L_K
        return J_L_Y

    def initialize_kernels(self, channels_in, input_size):
        self.input_size = input_size
        self.channels_in = channels_in
        self.kernels = np.random.uniform(self.weight_range[0], self.weight_range[1],
                                         (self.num_kernels, channels_in,) + self.kernel_size)

    def update_parameters(self, regularization=None):
        weight_gradient = self.weight_gradient
        batch_w_gradient = 1 / weight_gradient.shape[0] * weight_gradient
        if regularization:
            batch_w_gradient += regularization(self.kernels, derivative=True)
        self.kernels = self.kernels - self.learning_rate * batch_w_gradient


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

    def update_parameters(self, regularization=None):
        weight_gradient = self.weight_gradient
        batch_w_gradient = 1 / weight_gradient.shape[0] * np.sum(weight_gradient,
                                                                 axis=0)
        if regularization:
            batch_w_gradient += regularization(self.weights, derivative=True)
        self.weights = self.weights - self.learning_rate * batch_w_gradient

        # Calculate averaged bias gradient, add regularization term if applicable,
        # and update biases
        bias_gradient = self.bias_gradient
        batch_b_gradient = 1 / bias_gradient.shape[0] * np.sum(bias_gradient, axis=0)
        if regularization:
            batch_b_gradient += regularization(self.biases, derivative=True)

        self.biases = self.biases - self.learning_rate * batch_b_gradient



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


if __name__ == "__main__":
    '''
    layer = Conv1D(activation='linear', kernel_size=3, num_kernels=3, stride=2, mode='same')
    layer.initialize_kernels(2)
    test_data = np.array([[[1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]],
                          [[1, 0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 0, 0]]])
                          
    
    layer = Conv2D(activation='linear', kernel_size=(2, 2), num_kernels=3, stride=(1, 1), mode=('same', 'same'))
    layer.initialize_kernels(2)
    test_data = np.array([[[[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1]],
                           [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0]]],
                          [[[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0]],
                           [[1, 1, 1, 1], [1, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]]])
    out = layer.forward_pass(test_data)
    print(out)
    print(out.shape)
    print("___")
    print(layer.kernels)
    print(layer.kernels.shape)
    '''
