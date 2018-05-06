from tests.circle_classifier_test import CircleClassifierTest


test = CircleClassifierTest(4, 1000)
classifier = test.run()
train_f1 = classifier.compute_f1_score('train')

print('Train F1 Score: ' + str(train_f1))
