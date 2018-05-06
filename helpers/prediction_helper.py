import numpy as np
from services.data_preprocessor_service import DataPreprocessorService as dps


class PredictionHelper:

    @staticmethod
    def predict(Z):
        return Z.argmax(axis=1)
