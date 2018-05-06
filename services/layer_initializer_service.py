from models import *


class LayerInitializerService:

    @staticmethod
    def load_layers(num_classes: int, learning_rate: float):
        fc_layer_1 = fully_connected_layer_model.FullyConnectedLayerModel(67500, 50,
                                                                          'fc1', learning_rate)
        activation_layer_1 = activation_layer_model.ActivationLayerModel('relu', 'output_activation')
        output_fc = fully_connected_layer_model.FullyConnectedLayerModel(50, num_classes, 'fc2', learning_rate)
        output_activation = activation_layer_model.ActivationLayerModel('softmax', 'output_activation')

        # layers list
        layers = [
            fc_layer_1,
            activation_layer_1,
            output_fc,
            output_activation
        ]

        return layers
