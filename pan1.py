import torch
import torch.nn.functional as F
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
            GCNConv(hidden_channels, out_channels)
        ])

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# Load the dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
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

# Training loop with layer freezing/unfreezing
num_epochs_per_stage = 10
best_acc = 0
best_model_state = None
acc_history = []

for layer_index in range(5):

    set_requires_grad(model, layer_index + 1, requires_grad=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    print(f"Training with layers up to layer {layer_index+1} unfrozen")
    for epoch in range(num_epochs_per_stage):
        loss = train(model, data, optimizer)
        acc = evaluate(model, data)
        print(f'Epoch: {epoch + 1 + (layer_index + 1) * num_epochs_per_stage}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        # Compare performance and store the best model
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            best_layer_index = layer_index
        acc_history.append((layer_index, acc))
    acc1 = acc

    set_requires_grad(model, layer_index+2, requires_grad=True)
    print(f"Unfreezing layer {layer_index + 2} after initial training.")
    # Re-train with all layers up to the current layer frozen
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    for epoch in range((layer_index+2)*num_epochs_per_stage):
        loss = train(model, data, optimizer)
        acc = evaluate(model, data)
        print(f'Epoch: {epoch + 1 + (layer_index + 1) * num_epochs_per_stage}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        # Compare performance and store the best model
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            best_layer_index = layer_index
        acc_history.append((layer_index, acc))
    acc2 = acc

    set_requires_grad(model, layer_index+1, requires_grad=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    print(f"Re-freezing layer {layer_index + 1} to compare.")
    for epoch in range(num_epochs_per_stage):
        loss = train(model, data, optimizer)
        acc = evaluate(model, data)
        print(f'Epoch: {epoch + 1 + (layer_index + 1) * num_epochs_per_stage}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        # Compare performance and store the best model
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            best_layer_index = layer_index
        acc_history.append((layer_index, acc))
    acc_1 = acc
    if acc_1 > 2*acc2 and acc1 > 2*acc2:
        set_requires_grad(model, layer_index + 1, requires_grad=True)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
        # best_layer_index = layer_index
        print(f"Re-freezing layer {layer_index + 1} due to no improvement.")
        break

set_requires_grad(model, best_layer_index, requires_grad=True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
not_improve=0
for epoch in range(1500):
    loss = train(model, data, optimizer)
    acc = evaluate(model, data)
    print(f'Epoch: {epoch + 1 }, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

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
print(f'Best accuracy achieved: {final_acc:.4f} with layers up to layer {best_layer_index + 1} unfrozen.')
