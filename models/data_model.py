# model structure for the data to pass to classifier
from services.data_preprocessor_service import DataPreprocessorService as dps


class DataModel:

    def __init__(self, data: dict, num_classes: int, image_size: list):
        self.x_train = dps.preprocess_imagebatch(data['x_train'][0:500], image_size)
        self.y_train = dps.one_hot_encode(data['y_train'], num_classes)
