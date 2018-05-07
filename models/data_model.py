# model structure for the data to pass to classifier
from services.data_preprocessor_service import DataPreprocessorService as dps
import numpy as np


class DataModel:

    def __init__(self, data: dict, num_classes: int):
        self.x_train = data['x_train']
        self.y_train = dps.one_hot_encode(data['y_train'], num_classes)
