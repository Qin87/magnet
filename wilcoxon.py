import numpy as np
from scipy.stats import wilcoxon

# Sample data
# These arrays should contain the paired data
data1 = np.array([76.05, 76.09, 77.76, 73.68, 74.12, 74.26, 76.33, 77.26, 76.26, 75.44, 73.49, 77.87, 77.66, 76.87, 76.84, 76.43, 73.50, 70.61, 72.71, 71.62, 74.88, 76.98, 78.37, 73.35, 74.88, 75.96, 73.05, 73.34, 76.48, 74.51])
data2 = np.array([ 75.13, 77.04, 73.65, 73.88, 72.79, 70.29, 75.99, 73.18, 74.92, 66.99, 73.40, 75.01, 76.65, 73.29, 75.00, 73.13, 76.30, 74.43, 75.87, 74.16, 74.14, 75.92, 75.66, 73.26, 74.46, 74.96, 75.66, 75.77, 74.94, 75.08])

# Perform Wilcoxon signed-rank test
stat, p_value = wilcoxon(data1, data2)

print(f'Statistic: {stat}')
print(f'p-value: {p_value}')