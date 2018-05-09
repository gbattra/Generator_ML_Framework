# service to preprocesses images
import helpers.image_transform as it
from skimage.data import imread
import sklearn.utils as utils
from os import listdir
import numpy as np


class DataPreprocessorService:

    @staticmethod
    def load_data():
        data = dict(x_train=None, y_train=None, x_val=None, y_val=None)

        data['x_train'], data['y_train'] = DataPreprocessorService.load_datasets('train')

        return data

    @staticmethod
    def load_datasets(training_phase):
        imagesets = DataPreprocessorService.load_imagesets(training_phase)
        imageset = DataPreprocessorService.merge_imagesets(imagesets)
        shuffled_imageset = DataPreprocessorService.unison_shuffle_images_labels(imageset['x'], imageset['y'])

        return shuffled_imageset['x'], shuffled_imageset['y']

    @staticmethod
    def load_imagesets(data_type: str):
        imagesets = []
        directory = '../datasets/' + data_type
        for c, imageset in enumerate(listdir(directory)):
            c_images = []
            c_labels = []
            directory += '/' + imageset
            for image in listdir(directory):
                c_images.append(imread(directory + '/' + image))
                c_labels.append(c)

            # pack into dict
            c_image_data = {
                'x': c_images,
                'y': c_labels
            }

            imagesets.append(c_image_data)

            # reset directory
            directory = '../datasets/train'

        return imagesets

    @staticmethod
    def merge_imagesets(imagesets: list):
        merged_images = []
        merged_labels = []
        for imageset in imagesets:
            merged_images += imageset['x']
            merged_labels += imageset['y']

        return {
            'x': merged_images,
            'y': merged_labels
        }

    @staticmethod
    def preprocess_imagebatch(imagebatch, image_size: list):
        processed_imagebatch = np.zeros((len(imagebatch), image_size[0], image_size[1]))
        for i, image in enumerate(imagebatch):
            image = image[:, :]
            image = it.resize_image(image, image_size)

            processed_imagebatch[i, :, :] = image

        return processed_imagebatch.reshape(processed_imagebatch.shape[0], processed_imagebatch.shape[1], processed_imagebatch.shape[2], 1)

    @staticmethod
    def unison_shuffle_images_labels(images: list, labels: list):
        results = utils.shuffle(images, labels, random_state=np.random.randint(10))

        return {
            'x': results[0],
            'y': results[1]
        }

    @staticmethod
    def one_hot_encode(labels, num_classes):
        return np.eye(num_classes)[labels]
