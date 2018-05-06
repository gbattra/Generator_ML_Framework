from classifiers.shape_classifier import ShapeClassifier
from services.weight_initializer_service import DenseNNWeightInitializerService
import numpy as np


class ShapeGenerator:

    def __init__(self, shape_class: list, classifier: ShapeClassifier, image_shape: list, learning_rate: float = 0.01):
        self.shape_class = shape_class
        self.classifier = classifier
        self.image_shape = image_shape
        self.W, self.b = DenseNNWeightInitializerService.random_initialize_weights(self.image_shape)
        self.learning_rate = learning_rate

    def train(self, num_epochs):
        # starting image matrix
        inputs = np.ones((self.image_shape[0], self.image_shape[1]))

        # get classifier
        classifier = self.classifier

        for i in range(num_epochs):
            # generate initial image
            image = inputs * self.W + self.b
            classifier.y_pred = classifier.forward_propogate(image)
            grads = classifier.backward_propogate(self.shape_class)
            self.update_weights(grads)

    def update_weights(self, grads):
        update_param_W, update_param_b = grads['dW'], grads['db']

        self.W -= self.learning_rate * update_param_W
        self.b -= self.learning_rate * update_param_b

        return self
