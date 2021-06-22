import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

prediction_file = open("baseline_35_predictions_test.txt", "r")
predictions = prediction_file.read()
predictions = np.array([float(x) for x in predictions.split(", ")])
prediction_file.close()

ground_truth_file = open("baseline_35_ground_truth_test.txt", "r")
ground_truth = ground_truth_file.read()
ground_truth = np.array([float(x) for x in ground_truth.split(", ")])
ground_truth_file.close()

differences = ground_truth - predictions

sns.set_theme(style="whitegrid")
plt.figure(figsize=(17, 10))
sns.histplot(differences)
sns.despine()
#plt.title("Deviations in predictions, evaluated on test set")
plt.xlim(-1, 1)
plt.xlabel("Difference between ground truth and prediction")
plt.tight_layout()
plt.savefig("plots/prediction_histogram_v1.png", dpi=300)

plt.show()
