import numpy as np


class ShapeClassesHelper:

    @staticmethod
    def get_class_for_shape(shape: str):
        classes = np.eye(4)
        shape_classes = {
            'circle': classes[0],
            'square': classes[1],
            'star': classes[2],
            'triangle': classes[3]
        }

        return shape_classes[shape]

    @staticmethod
    def get_shape_for_class(prediction):
        classes = np.eye(4)
        shape_classes = {
            'circle': classes[0],
            'square': classes[1],
            'star': classes[2],
            'triangle': classes[3]
        }
        for shape in shape_classes:
            if np.asmatrix(shape_classes[shape]).argmax(axis=1) == prediction:
                return shape
