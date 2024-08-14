import numpy as np
from scipy.stats import wilcoxon

# Sample data
# These arrays should contain the paired data
data1 = np.array([66.31, 67.68, 69.39, 68.35, 69.87, 67.98, 69.58, 67.05, 70.28, 66.20])
data2 = np.array([64.26, 69.09, 70.88, 67.57, 69.95, 68.57, 67.31, 65.42, 69.84, 66.01])

# Perform Wilcoxon signed-rank test
stat, p_value = wilcoxon(data1, data2)

print(f'Statistic: {stat}')
print(f'p-value: {p_value}')