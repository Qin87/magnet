import matplotlib.pyplot as plt
import numpy as np

# Data
layers = np.arange(1, 21)
accuracy = [65.3, 63.9, 61.1, 62.1, 57.8, 50.1, 41.6, 29.6, 26.8, 23.6, 22.5, 21.4, 22.0, 21.2, 21.5, 22.6, 20.5, 21.3, 21.6, 20.7]
std_dev = [1.7, 2.6, 2.4, 2.9, 3.9, 11.9, 12.3, 5.6, 7.0, 5.4, 3.7, 2.9, 2.6, 1.6, 2.2, 4.0, 1.6, 1.5, 2.1, 2.2]

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(layers, accuracy, yerr=std_dev, fmt='-o', capsize=5, label="Accuracy")
plt.title("Accuracy vs. Layer Depth")
plt.xlabel("Layer Depth")
plt.ylabel("Accuracy (%)")
plt.xticks(layers)
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()

# Save or show
plt.savefig("accuracy_vs_layer_depth.png", dpi=300)  # Save for ICML paper
plt.show()
