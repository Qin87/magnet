import matplotlib.pyplot as plt
import numpy as np

data = 'chame'
if data == 'chame':
    data_name = 'Chameleon'
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70]
    # k_acc_mean = [77.7, 78.1, 76.8, 75.8, 75.2, 74.2, 72.7, 71.8, 71.8, 70.2, 69.9, 69.5, 70.3, 69.5, 69.4, 69.8, 70.1, 69.8, 69.5, 69.4, 69.2, 69.9, 69.3, 70.8, 70.4]
    # k_acc_vari = [1.0, 1.5, 1.3, 2.3, 2.4, 2.2, 2.5, 2.0, 2.2, 3.2, 2.1, 3.2, 2.2, 2.9, 3.5, 2.8, 2.8, 2.4, 2.6, 1.8, 3.7, 2.2, 3.3, 1.4, 2.8]
    #
    acc_mean_1 = [75.7, 76.9, 76.1, 74.5, 73.0, 71.1, 68.9, 65.3, 63.8,
                  62.2, 58.0, 57.4, 58.1, 53.3, 52.5, 52.5, 52.9, 50.2, 52.5, 51.2, 52.5, 52.8, 50.7, 51.7, 51.4]
    acc_vari_1 = [1.3, 1.1, 1.3, 1.8, 2.0, 2.3, 2.2, 3.0, 3.8,
                  3.0, 4.8, 4.5, 2.0, 3.8, 5.7, 5.2, 5.4, 4.1, 3.4, 3.5, 3.0, 2.8, 2.9, 4.0, 4.9]

elif data == 'squi':
    data_name = 'Squirrel'
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50]  # 60-OOM
    k_acc_mean = []
    k_acc_vari = []

    acc_mean_1 = []
    acc_vari_1 = []

else:
    pass



# Plot
plt.figure(figsize=(15, 6))
plt.errorbar(layers, k_acc_mean, yerr=k_acc_vari, fmt='-o', capsize=4, label='1&2-hop')
plt.errorbar(layers, acc_mean_1, yerr=acc_vari_1, fmt='o-', capsize=4, label='1-hop')

# Labels and title
plt.title(data_name+ ' Accuracy vs Layers: Without Relu', fontsize=16)
plt.xlabel('Layer', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(layers, rotation=45)
plt.tight_layout()

plt.savefig(data_name+ " accuracy_multiple hop.pdf", dpi=300)
# Show plot
plt.show()
