# the model for an activation layer
import numpy as np


class ActivationLayerModel:

    def __init__(self,
                 activation: str,
                 name: str):
        self.activation = activation
        self.forward_cache = {}
        self.backward_cache = {}
        self.name = name

    def forward_propogate(self, A_prev):
        if self.activation == 'relu':
            A_activated = self.relu_activation(A_prev)
        elif self.activation == 'sigmoid':
            A_activated = self.sigmoid_activation(A_prev)
        elif self.activation == 'tanh':
            A_activated = self.tanh_activation(A_prev)
        elif self.activation == 'softmax':
            A_activated = self.softmax_activation(A_prev)
        else:
            A_activated = A_prev

        self.forward_cache = {
            'A_prev': A_prev,
            'A': A_activated
        }

        return A_activated

    def backward_propogate(self, grads, lamda: int, for_generator: bool=False):
        dZ_next = grads['dZ']
        dZ = dZ_next.T * self.get_derivative(self.activation, self.forward_cache['A'])

        self.backward_cache = {
            'dZ': dZ
        }

        if for_generator:
            dZ = self.compute_shirking_gradient(dZ_next, dZ, 'mean')

        return {
            'dZ': dZ
        }

    def get_derivative(self, activation_function, x):
        if activation_function == 'softmax':
            return self.softmax_derivative(x)
        elif activation_function == 'relu':
            return self.relu_derivative(x)
        elif activation_function == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif activation_function == 'tanh':
            return self.tanh_derivative(x)

    def compute_shirking_gradient(self, dZ_next, dZ, method: str):
        if method == 'mean':
            m = np.asmatrix(dZ_next.mean(axis=1))
            dZ += m

        return dZ

    def update_weights(self):
        return self  # activation layers have no weights to update

    def store_weights(self):
        return self  # no weights to save

    @staticmethod
    def tanh_derivative(x):
        return 1.0 - np.tanh(x) ** 2

    @staticmethod
    def relu_activation(x):
        return np.maximum(x, 0)

    @staticmethod
    def sigmoid_activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh_activation(x):
        return np.tanh(x)

    @staticmethod
    def softmax_activation(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    @staticmethod
    def relu_derivative(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
