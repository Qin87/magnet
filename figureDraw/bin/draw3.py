import matplotlib.pyplot as plt
import numpy as np

# Data for all three experiments
layers = list(range(1, 21))

# First experiment data (Skip Connection)
skip_Ak = [63.0, 59.9, 58.3, 58.6, 58.5, 58.6, 58.5, 58.5, 58.5, 58.5, 58.5, 58.5, 58.5, 58.5, 58.5, 58.5, 58.5, 58.5, 58.5, 58.5]
skip_layer = [63.1, 60.5, 59.5, 57.1, 54.8, 54.1, 51.7, 53.1, 47.3, 43.9, 40.9, 40.0, 36.1, 30.1, 25.2, 29.6, 22.5, 23.6, 22.8, 22.3]
skip_Ak_std = np.sqrt([1.9, 3.4, 2.1, 2.0, 2.0, 1.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
skip_layer_std = np.sqrt([2.0, 2.6, 2.3, 2.6, 3.8, 3.4, 3.8, 2.9, 3.6, 2.8, 2.6, 3.3, 7.1, 6.4, 5.9, 7.2, 3.3, 4.5, 2.0, 1.9])

# Second experiment data (Power Law)
power_Ak = [59.6, 57.8, 56.8, 56.5, 56.3, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4]
power_layer = [59.6, 56.7, 56.0, 53.7, 51.8, 50.2, 49.7, 47.3, 43.8, 38.5, 32.5, 34.6, 31.9, 25.7, 25.3, 21.0, 19.8, 19.6, 19.9, 19.7]
power_Ak_std = np.sqrt([2.4, 3.7, 3.2, 2.8, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9])
power_layer_std = np.sqrt([2.2, 1.4, 1.9, 2.9, 2.3, 2.7, 2.9, 3.4, 6.3, 4.6, 3.4, 4.5, 6.2, 6.0, 7.4, 4.0, 1.0, 1.5, 1.2, 1.5])

# Third experiment data (Higher Power)
high_Ak = [65.4, 66.0, 65.2, 65.1, 64.7, 63.7, 62.0, 60.1, 58.1, 56.0, 53.5, 50.4, 45.8, 41.4, 37.4, 35.7, 35.3, 35.0, 34.7, 35.0]
high_layer = [59.6, 56.5, 55.7, 54.1, 50.6, 51.1, 49.0, 48.6, 44.3, 35.8, 32.9, 33.5, 31.2, 26.8, 24.3, 22.9, 20.2, 19.7, 19.6, 19.4]
high_Ak_std = np.sqrt([3.2, 2.2, 0.8, 0.9, 1.0, 1.2, 1.4, 2.6, 1.7, 1.2, 1.6, 3.3, 6.6, 6.7, 5.7, 4.0, 3.6, 2.6, 2.0, 2.1])
high_layer_std = np.sqrt([2.4, 1.7, 2.1, 2.4, 3.2, 2.7, 4.4, 3.8, 4.1, 4.6, 6.0, 4.9, 5.8, 5.7, 5.4, 6.5, 1.7, 0.9, 1.2, 1.4])

# Create figure
plt.figure(figsize=(12, 6))

# Plot all experiments
plt.errorbar(layers, skip_Ak, yerr=skip_Ak_std, fmt='ro-', label='Skip Ak', capsize=3, markersize=5)
plt.errorbar(layers, skip_layer, yerr=skip_layer_std, fmt='ro--', label='Skip Layer', capsize=3, markersize=5, alpha=0.5)

plt.errorbar(layers, power_Ak, yerr=power_Ak_std, fmt='bo-', label='Power Ak', capsize=3, markersize=5)
plt.errorbar(layers, power_layer, yerr=power_layer_std, fmt='bo--', label='Power Layer', capsize=3, markersize=5, alpha=0.5)

plt.errorbar(layers, high_Ak, yerr=high_Ak_std, fmt='go-', label='High Power Ak', capsize=3, markersize=5)
plt.errorbar(layers, high_layer, yerr=high_layer_std, fmt='go--', label='High Power Layer', capsize=3, markersize=5, alpha=0.5)

# Set labels and title
plt.xlabel('Number of Layers')
plt.ylabel('Accuracy (%)')
plt.title('Comprehensive GCN Analysis: All Experiments')

# Add grid
plt.grid(True, alpha=0.3)

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('gcn_analysis_all_in_one.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()