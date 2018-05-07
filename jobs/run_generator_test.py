from tests.circle_classifier_test import CircleClassifierTest
from generators.shape_generator import ShapeGenerator
from helpers.shape_classes_helper import ShapeClassesHelper
import matplotlib.pyplot as plt
from models.activation_layer_model import ActivationLayerModel
from helpers.prediction_helper import PredictionHelper


test = CircleClassifierTest(4, 1000)
classifier = test.run()
shape_class = ShapeClassesHelper.get_class_for_shape('triangle')
activation = ActivationLayerModel('tanh', 'input_activation')
generator = ShapeGenerator(shape_class, classifier, activation, [100, 100])
generator.train(1000)
image = generator.generate_shape()
prediction = PredictionHelper.predict(classifier.y_pred)
class_name = ShapeClassesHelper.get_shape_for_class(prediction)
plt.imshow(image)
plt.title(class_name)
plt.show()
