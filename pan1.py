import os
from datetime import datetime

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from plotly.figure_factory import np
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Define the GCN model with 5 layers
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(in_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, out_channels)
        ])

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# Load the dataset
dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
data = dataset[0]

# Instantiate the model
model = GCN(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes)

# Function to freeze/unfreeze layers
def set_requires_grad(model, layer_index, requires_grad):
    for i, conv in enumerate(model.convs):
        for param in conv.parameters():
            param.requires_grad = requires_grad if i <= layer_index else False

# Train the model
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluate the model
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
    return acc

def append_value(layer_index, value):
    key = f'Layer {layer_index}'
    if key in acc_data:
        acc_data[key].append(value)
    else:
        print(f"Layer {layer_index} does not exist in the data structure.")

# Training loop with layer freezing/unfreezing
num_epochs_per_stage = 200
best_acc = 0
best_model_state = None
acc_history = []

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
pan_path = os.path.expanduser('/home/qin/Documents/Benlogs/')
log_file_name = 'GCN_CiteSeer_' + str(num_epochs_per_stage)
log_file_name_with_timestamp = pan_path+ log_file_name +'_T' + timestamp+'.log'
acc_data = {f'Layer {i+1}': [] for i in range(8)}
append_value(1, 0)


with open(log_file_name_with_timestamp, 'w+') as log_file:
    log_file.write("Test log message\n")

# Check if the file has been created
print(f"File '{log_file_name_with_timestamp}' has been created.")

with open(log_file_name_with_timestamp, 'w+') as log_file:

    for layer_index in range(1, 9):
        # (1) original layer
        set_requires_grad(model, layer_index, requires_grad=True)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80, verbose=True)
        print(f"Training with layers up to layer {layer_index} unfrozen")
        print(f"Training with layers up to layer {layer_index} unfrozen", file=log_file)
        for epoch in range(num_epochs_per_stage):
            loss = train(model, data, optimizer)
            acc = evaluate(model, data)
            # print(f'Epoch: {epoch + 1 + (layer_index + 1) * num_epochs_per_stage}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

            # Compare performance and store the best model
            if acc > best_acc:
                best_acc = acc
                best_model_state = model.state_dict()
                best_layer_index = layer_index

            acc_history.append((layer_index, acc))
        print('best layer index is ' + str(best_layer_index), file=log_file)
        print('best layer index is ' + str(best_layer_index))
        acc1 = acc
        print(acc)
        print(acc, file=log_file)
        append_value(layer_index, acc)

        # (2) unfreeze original lay+1
        set_requires_grad(model, layer_index+1, requires_grad=True)
        print(f"Unfreezing layer {layer_index + 1} after initial training.")
        print(f"Unfreezing layer {layer_index + 1} after initial training.", file=log_file)
        # Re-train with all layers up to the current layer frozen
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80, verbose=True)
        for epoch in range((layer_index+1)*num_epochs_per_stage):
            loss = train(model, data, optimizer)
            acc = evaluate(model, data)
            # print(f'Epoch: {epoch + 1 + (layer_index + 1) * num_epochs_per_stage}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

            # Compare performance and store the best model
            if acc > best_acc:
                best_acc = acc
                best_model_state = model.state_dict()
                best_layer_index = layer_index+1

            acc_history.append((layer_index, acc))
        print('best layer index is ' + str(best_layer_index), file=log_file)
        print('best layer index is ' + str(best_layer_index))
        acc2 = acc
        print(acc)
        print(acc, file=log_file)
        append_value(layer_index+1, acc)

        # (3) back to original layer
        set_requires_grad(model, layer_index, requires_grad=True)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80, verbose=True)
        print(f"Re-freezing layer {layer_index } to compare.")
        print(f"Re-freezing layer {layer_index } to compare.", file=log_file)
        for epoch in range(num_epochs_per_stage):
            loss = train(model, data, optimizer)
            acc = evaluate(model, data)
            # print(f'Epoch: {epoch + 1 + (layer_index + 1) * num_epochs_per_stage}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

            # Compare performance and store the best model
            if acc > best_acc:
                best_acc = acc
                best_model_state = model.state_dict()
                best_layer_index = layer_index

            acc_history.append((layer_index, acc))
        print('best layer index is ' + str(best_layer_index), file=log_file)
        print('best layer index is ' + str(best_layer_index))
        acc_1 = acc
        print(acc)
        print(acc, file=log_file)
        append_value(layer_index, acc)
        if acc_1 > 1.5*acc2 and acc1 > 1.5*acc2:
            set_requires_grad(model, layer_index + 1, requires_grad=True)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80, verbose=True)
            # best_layer_index = layer_index
            print(f"Re-freezing layer {layer_index + 1} due to no improvement.")
            print(f"Re-freezing layer {layer_index + 1} due to no improvement.", file=log_file)
            break

    set_requires_grad(model, best_layer_index, requires_grad=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80, verbose=True)
    not_improve=0
    for epoch in range(1500):
        loss = train(model, data, optimizer)
        acc = evaluate(model, data)
        # print(f'Epoch: {epoch + 1 }, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        # Compare performance and store the best model
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            # best_layer_index = layer_index
        else:
            not_improve += 1
        if not_improve> 110:
            break

    # Load the best model state
    model.load_state_dict(best_model_state)
    final_acc = evaluate(model, data)
    print(f'Best accuracy achieved: {final_acc:.4f} with layers up to layer {best_layer_index } unfrozen.')
    print(f'Best accuracy achieved: {final_acc:.4f} with layers up to layer {best_layer_index } unfrozen.', file=log_file)

layers = list(acc_data.keys())
values = np.array(list(acc_data.values()))

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data
width = 0.2  # Width of each bar
x = np.arange(len(layers))  # X axis locations for the groups

# Plot each set of numbers for each layer
for i in range(values.shape[1]):
    ax.bar(x + i * width, values[:, i], width, label=f'Number {i+1}')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Layers')
ax.set_ylabel('Accuracy')
ax.set_title(log_file_name)
ax.set_xticks(x + width)
ax.set_xticklabels(layers)
ax.legend()

# Save the plot as a file
plt.savefig(log_file_name+'_'+timestamp+'.png')

# Show the plot (optional)
plt.show()