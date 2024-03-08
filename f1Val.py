import numpy as np

# Assuming the log is stored in a list of dictionaries called 'results'
results = [
    {"acc": 72.97, "bacc": 52.67, "f1": 52.28},
    {"acc": 64.86, "bacc": 55.83, "f1": 57.93},
    {"acc": 70.27, "bacc": 70.00, "f1": 61.17},
    {"acc": 75.68, "bacc": 57.38, "f1": 63.52},
    {"acc": 72.97, "bacc": 50.00, "f1": 46.23},
    {"acc": 67.57, "bacc": 45.04, "f1": 51.35},
    {"acc": 72.97, "bacc": 49.38, "f1": 46.47},
    {"acc": 67.57, "bacc": 48.35, "f1": 49.24},
    {"acc": 48.65, "bacc": 52.75, "f1": 52.53},
    {"acc": 64.86, "bacc": 45.06, "f1": 41.20}
]

# Extract acc, bacc, and f1 from each split
acc_values = [result["acc"] for result in results]
bacc_values = [result["bacc"] for result in results]
f1_values = [result["f1"] for result in results]

# Compute mean and variance
acc_mean = np.mean(acc_values)
acc_variance = np.var(acc_values)
bacc_mean = np.mean(bacc_values)
bacc_variance = np.var(bacc_values)
f1_mean = np.mean(f1_values)
f1_variance = np.var(f1_values)

# Print mean and variance
print("Mean Accuracy:", acc_mean)
print("Variance Accuracy:", acc_variance)
print("Mean Balanced Accuracy:", bacc_mean)
print("Variance Balanced Accuracy:", bacc_variance)
print("Mean F1 Score:", f1_mean)
print("Variance F1 Score:", f1_variance)
