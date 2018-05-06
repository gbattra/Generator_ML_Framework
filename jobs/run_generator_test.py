from tests.circle_classifier_test import CircleClassifierTest
from generators.shape_generator import ShapeGenerator
from helpers.shape_classes_helper import ShapeClassesHelper
import matplotlib.pyplot as plt


test = CircleClassifierTest(4, 1000)
classifier = test.run()
shape_class = ShapeClassesHelper.get_class_for_shape('circle')
generator = ShapeGenerator(shape_class, classifier, [150, 150])
generator.train(10000)
image = generator.generate_shape()
plt.imshow(image)
plt.title('Circle')
plt.show()
