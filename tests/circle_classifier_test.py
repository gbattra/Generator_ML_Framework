# the test process for the spongebob character classifier

from services.data_preprocessor_service import DataPreprocessorService
from classifiers.classifier import Classifier
from models import data_model, fully_connected_layer_model, activation_layer_model


class CircleClassifierTest:

    def __init__(self, num_classes: int, num_iters: int):
        data = {
            'x_train': None,
            'y_train': None,
            'x_val': None,
            'y_val': None
        }
        self.num_iters = num_iters
        self.num_classes = num_classes

        data['x_train'], data['y_train'] = self.load_imagesets('train')
        self.data_model = data_model.DataModel(data, num_classes, [100, 100])
        self.learning_rate = 0.01

    def run(self):
        fc_layer_1 = fully_connected_layer_model.FullyConnectedLayerModel(30000, 10, 'fc1', self.learning_rate)
        activation_layer_1 = activation_layer_model.ActivationLayerModel('relu', 'output_activation')
        output_fc = fully_connected_layer_model.FullyConnectedLayerModel(10, self.num_classes, 'fc2', self.learning_rate)
        output_activation = activation_layer_model.ActivationLayerModel('softmax', 'output_activation')

        # layers list
        layers = [
            fc_layer_1,
            activation_layer_1,
            output_fc,
            output_activation
        ]

        # instantiate classifier model
        classifier_model = Classifier(self.data_model, self.num_iters, layers, 0.1)

        # train model
        classifier_model.train(classifier_model.data.x_train, classifier_model.data.y_train)

        return classifier_model

    def load_imagesets(self, training_phase):
        imagesets = DataPreprocessorService.load_imagesets(training_phase)
        imageset = DataPreprocessorService.merge_imagesets(imagesets)
        shuffled_imageset = DataPreprocessorService.unison_shuffle_images_labels(imageset['x'], imageset['y'])

        return shuffled_imageset['x'], shuffled_imageset['y']
