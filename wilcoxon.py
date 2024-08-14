import numpy as np
from scipy.stats import wilcoxon

# Sample data
# These arrays should contain the paired data
data1 = np.array([98, 94.00, 98.00, 94.00, 98.00, 100.00, 94.00, 96.00, 98.00, 98.00, 90.00, 94.00, 96.00, 100.00, 98.00, 92.00, 96.00, 96.00, 94.00, 94.00, 98.00, 90.00, 100.00, 100.00, 100.00, 90.00, 94.00, 94.00, 96.00, 98.00])
data2 = np.array([100.00, 96.00, 92.00, 92.00, 98.00, 100.00, 90.00, 96.00, 96.00, 92.00, 90.00, 94.00, 94.00, 98.00, 88.00, 96.00, 82.00, 90.00, 96.00, 90.00, 92.00, 94.00, 98.00, 88.00, 96.00, 98.00, 84.00, 90.00, 94.00, 94.00])

# Perform Wilcoxon signed-rank test
stat, p_value = wilcoxon(data1, data2)

print(f'Statistic: {stat}')
print(f'p-value: {p_value}')