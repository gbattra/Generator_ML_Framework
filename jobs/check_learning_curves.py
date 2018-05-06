from services.learning_curves_service import LearningCurvesService
import matplotlib.pyplot as plt


learning_curves = LearningCurvesService.compute_learning_curves(10, 5)

plt.plot(range(1, len(learning_curves['train_cost']) + 1), learning_curves['train_cost'])
plt.plot(range(1, len(learning_curves['val_cost']) + 1), learning_curves['val_cost'])
plt.show()
