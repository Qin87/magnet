import matplotlib.pyplot as plt
import numpy as np

data = 'Squirrel'
# if data == 'chame':
data_name1 = 'Chameleon'
layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50]
# k_acc_mean = [77.7, 78.1, 76.8, 75.8, 75.2, 74.2, 72.7, 71.8, 71.8, 70.2, 69.9, 69.5, 70.3, 69.5, 69.4, 69.8, 70.1, 69.8, 69.5, 69.4, 69.2, 69.9, 69.3, 70.8, 70.4]
# k_acc_vari = [1.0, 1.5, 1.3, 2.3, 2.4, 2.2, 2.5, 2.0, 2.2, 3.2, 2.1, 3.2, 2.2, 2.9, 3.5, 2.8, 2.8, 2.4, 2.6, 1.8, 3.7, 2.2, 3.3, 1.4, 2.8]
#
acc_mean_1_chame = [75.7, 76.8, 77.0, 75.3, 75.2, 74.7, 74.5, 74.3, 74.3, 74.1,
                    73.3, 73.1, 73.3, 73.7, 74.3, 72.8, 73.4, 73.1, 73.8, 73.9, 73.3, 74.2, 73.8]
acc_vari_1_chame = [1.3, 1.2, 1.4, 1.3, 2.3, 2.3, 2.1, 2.2, 2.2, 1.7,
                    2.1, 1.9, 2.5, 3.4, 2.6, 3.4, 2.4, 3.3, 2.5, 2.1, 2.4, 2.4, 2.9]

# elif data == 'squi':
data_name2 = 'Squirrel'
layers_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50]  # 60-OOM
# k_acc_mean = []
# k_acc_vari = []

acc_mean_1_squi = [73.2, 72.4, 73.0, 72.9, 73.0, 73.0, 73.3, 73.0, 73.1, 73.1,
              72.7, 72.7, 73.2, 73.5, 72.8, 72.9, 73.2, 73.5, 72.8, 72.9, 73.0, 72.3, 72.7]
acc_vari_1_squi = [1.8, 1.8, 2.0, 1.4, 1.9, 1.9, 1.9, 1.6, 1.7, 2.1,
              1.8, 2.3, 1.8, 1.5, 1.6, 1.8, 1.6, 1.8, 2.0, 1.9, 1.4, 3.4, 1.6]

# else:
#     pass



# Plot
plt.figure(figsize=(15, 6))
# plt.errorbar(layers, k_acc_mean, yerr=k_acc_vari, fmt='-o', capsize=4, label='1&2-hop')
# plt.errorbar(layers, acc_mean_1, yerr=acc_vari_1, fmt='o-', capsize=4, label='1-hop')

# Labels and title
plt.title( ' Accuracy vs Layers: Without Relu', fontsize=16)
plt.xlabel('Layer', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(layers, rotation=45)
plt.tight_layout()

plt.savefig(data_name+ " accuracy_multiple hop.pdf", dpi=300)
# Show plot
plt.show()
