import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid


# Define the GCN model with 5 layers
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([
            GCNConv(in_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, out_channels)
        ])

    def forward(self, x, edge_index):
        out_list = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
            out_list.append(F.log_softmax(x, dim=1))
        return out_list


# Load the dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Instantiate the model with 5 layers
model_5_layers = GCN(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes, num_layers=5)

# Instantiate the model with 3 layers
model_3_layers = GCN(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes, num_layers=3)


# Train the model
def train(model, data, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out_list = model(data.x, data.edge_index)
        loss = sum(F.nll_loss(out[data.train_mask], data.y[data.train_mask]) for out in out_list)
        loss.backward()
        optimizer.step()


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out_list = model(data.x, data.edge_index)
        acc_list = [evaluate_single(out, data) for out in out_list]
    return acc_list


def evaluate_single(out, data):
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc


# Initialize optimizer for 5-layer model
optimizer_5_layers = torch.optim.Adam(model_5_layers.parameters(), lr=0.01)

# Initialize optimizer for 3-layer model
optimizer_3_layers = torch.optim.Adam(model_3_layers.parameters(), lr=0.01)

# Define parameters for training stages
num_epochs_per_stage = 10
best_acc_5_layers = 0
best_model_state_5_layers = None
best_layer_index_5_layers = -1

best_acc_3_layers = 0
best_model_state_3_layers = None
best_layer_index_3_layers = -1

# Training loop with evaluation at the end of each stage for 5-layer model
for layer_index in range(1, 6):  # From layer 1 to layer 5
    # Unfreeze up to the current layer
    for i, conv in enumerate(model_5_layers.convs):
        requires_grad = i < layer_index
        for param in conv.parameters():
            param.requires_grad = requires_grad

    # Optimizer for the current set of trainable parameters
    optimizer_5_layers = torch.optim.Adam(filter(lambda p: p.requires_grad, model_5_layers.parameters()), lr=0.01)

    print(f"Training with layers up to layer {layer_index} unfrozen for 5-layer model.")

    # Training loop for num_epochs_per_stage epochs for 5-layer model
    for epoch in range(num_epochs_per_stage):
        train(model_5_layers, data, optimizer_5_layers, num_epochs=1)  # Train for 1 epoch
        acc_list = evaluate(model_5_layers, data)
        acc_str = ", ".join([f"{acc:.4f}" for acc in acc_list])
        print(f'Epoch: {epoch + 1 + (layer_index - 1) * num_epochs_per_stage}, Accuracy: [{acc_str}]')

        # Compare performance and store the best model for 5-layer model
        if acc_list[-1] > best_acc_5_layers:
            best_acc_5_layers = acc_list[-1]
            best_model_state_5_layers = model_5_layers.state_dict()
            best_layer_index_5_layers = layer_index

# Training loop with evaluation at the end of each stage for 3-layer model
for layer_index in range(1, 4):  # From layer 1 to layer 3
    # Unfreeze up to the current layer
    for i, conv in enumerate(model_3_layers.convs):
        requires_grad = i < layer_index
        for param in conv.parameters():
            param.requires_grad = requires_grad

    # Optimizer for the current set of trainable parameters
    optimizer_3_layers = torch.optim.Adam(filter(lambda p: p.requires_grad, model_3_layers.parameters()), lr=0.01)

    print(f"Training with layers up to layer {layer_index} unfrozen for 3-layer model.")

    # Training loop for num_epochs_per_stage epochs for 3-layer model
    for epoch in range(num_epochs_per_stage):
        train(model_3_layers, data, optimizer_3_layers, num_epochs=1)  # Train for 1 epoch
        acc_list = evaluate(model_3_layers, data)
        acc_str = ", ".join([f"{acc:.4f}" for acc in acc_list])
        print(f'Epoch: {epoch + 1 + (layer_index - 1) * num_epochs_per_stage}, Accuracy: [{acc_str}]')

        # Compare performance and store the best model for 3-layer model
        if acc_list[-1] > best_acc_3_layers:
            best_acc_3_layers = acc_list[-1]
            best_model_state_3_layers = model_3_layers.state_dict()
            best_layer_index_3_layers = layer_index

# Print the best accuracy achieved for 5-layer model
print(f'Best accuracy achieved for 5-layer model: {best_acc_5_layers:.4f} with layers up to layer {best_layer_index_5_layers} unfrozen.')

# Print the best accuracy achieved for 3-layer model
print(f'Best accuracy achieved for 3-layer model: {best_acc_3_layers:.4f} with layers up to layer {best_layer_index_3_layers} unfrozen.')
