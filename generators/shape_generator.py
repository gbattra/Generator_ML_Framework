from classifiers.shape_classifier import ShapeClassifier
from services.weight_initializer_service import DenseNNWeightInitializerService
import numpy as np


class ShapeGenerator:

    def __init__(self, shape_class: list, classifier: ShapeClassifier, image_shape: list, learning_rate: float = 0.000001):
        self.shape_class = shape_class
        self.classifier = classifier
        self.image_shape = image_shape
        self.W, self.b = DenseNNWeightInitializerService.random_initialize_weights(self.image_shape)
        self.learning_rate = learning_rate
        self.grads = {}

    def train(self, num_epochs):
        # starting image matrix
        inputs = np.ones((1, self.image_shape[0], self.image_shape[1]))

        # get classifier
        classifier = self.classifier

        for i in range(num_epochs):
            # generate initial image
            image = inputs * self.W + self.b
            image.reshape(1, image.size)
            classifier.y_pred = classifier.forward_propogate(image)
            grads = classifier.backward_propogate(self.shape_class, True)
            cost = classifier.compute_cost(self.shape_class, classifier.y_pred, False)
            print(cost)
            self.backward_propogate(inputs, grads)
            self.update_weights()

    def update_weights(self):
        update_param_W, update_param_b = self.grads['dW'], self.grads['db']

        self.W -= self.learning_rate * update_param_W
        self.b -= self.learning_rate * update_param_b

        return self

    def backward_propogate(self, inputs, grads):
        dZ = grads['dZ']
        dZ = dZ.reshape(150, 150)
        dW = dZ
        db = np.sum(dZ)

        self.grads = {
            'dW': dW,
            'db': db
        }

        return self

    def generate_shape(self):
        inputs = np.ones((self.image_shape[0], self.image_shape[1]))
        image = inputs * self.W + self.b

        return image
