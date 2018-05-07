# the test process for the circle classifier
from models.conv_filter_model import CONVFilterModel
from models.conv_layer_model import CONVLayerModel
from services.data_preprocessor_service import DataPreprocessorService
from classifiers.shape_classifier import ShapeClassifier
from models import data_model, fully_connected_layer_model, activation_layer_model


class CircleClassifierTest:

    def __init__(self, num_classes: int, num_iters: int):
        self.num_iters = num_iters
        self.num_classes = num_classes

        data = DataPreprocessorService.load_data()
        self.data_model = data_model.DataModel(data, num_classes)
        self.learning_rate = 0.01

    def run(self):
        conv_filter_1 = CONVFilterModel(4, 3, 8)
        conv_layer_1 = CONVLayerModel(conv_filter_1, [2, 2], 'same', 'tanh_1')
        # relu layer 1
        tanh_layer_1 = activation_layer_model.ActivationLayerModel('tanh', 'tanh_1')

        fc_layer_1 = fully_connected_layer_model.FullyConnectedLayerModel(10000, 10, 'fc1', self.learning_rate)
        activation_layer_1 = activation_layer_model.ActivationLayerModel('relu', 'output_activation')

        output_fc = fully_connected_layer_model.FullyConnectedLayerModel(10, self.num_classes, 'fc2', self.learning_rate)
        output_activation = activation_layer_model.ActivationLayerModel('softmax', 'output_activation')

        # layers list
        layers = [
            # conv_layer_1,
            # tanh_layer_1,
            fc_layer_1,
            activation_layer_1,
            output_fc,
            output_activation
        ]

        # instantiate classifier model
        classifier_model = ShapeClassifier(self.data_model, self.num_iters, layers, 0.1)

        # train model
        # classifier_model.train(classifier_model.data.x_train, classifier_model.data.y_train)

        return classifier_model
