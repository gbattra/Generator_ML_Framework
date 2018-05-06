from classifiers.classifier import SpongebobCharacterClassifier
from services.data_preprocessor_service import DataPreprocessorService as dps
from services.layer_initializer_service import LayerInitializerService
from models.data_model import DataModel


class LearningCurvesService:

    @staticmethod
    def compute_learning_curves(batch_size, lamda):
        data = dps.load_data()
        data_model = DataModel(data, 7, [100, 100])

        costs = {'train_cost': [], 'val_cost': []}
        for i in range(int(len(data_model.x_train) / batch_size)):
            end = (i + 1) * batch_size

            layers = LayerInitializerService.load_layers(7, 0.01)
            classifier = SpongebobCharacterClassifier(data_model, 1000, layers, lamda)
            x_train, y_train = classifier.data.x_train, classifier.data.y_train
            x_val, y_val = classifier.data.x_val, classifier.data.y_val

            # train on train data
            x_train_batch = x_train[0:end]
            y_train_batch = y_train[0:end]
            print(x_train_batch.shape)

            classifier.train(x_train_batch, y_train_batch)
            train_cost = classifier.compute_cost(y_train_batch, classifier.y_pred, False)

            # get cost on cv data
            y_val_pred = classifier.forward_propogate(x_val)
            cv_cost = classifier.compute_cost(y_val, y_val_pred, False)

            print(train_cost)
            costs['train_cost'].append(train_cost)
            costs['val_cost'].append(cv_cost)

        return costs
