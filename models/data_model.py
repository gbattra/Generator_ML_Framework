# model structure for the data to pass to classifier
from services.data_preprocessor_service import DataPreprocessorService as dps


class DataModel:

    def __init__(self, data: dict, num_classes: int, image_size: list):
        self.x_train = dps.preprocess_imageset(data['x_train'], image_size)
        self.y_train = dps.one_hot_encode(data['y_train'], num_classes)
        self.x_val = dps.preprocess_imageset(data['x_val'], image_size)
        self.y_val = dps.one_hot_encode(data['y_val'], num_classes)
        self.x_test = dps.preprocess_imageset(data['x_test'], image_size)
        self.y_test = dps.one_hot_encode(data['y_test'], num_classes)
