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
