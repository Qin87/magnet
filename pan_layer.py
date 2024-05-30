import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader


# Define the GCN model with 5 layers
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        return F.log_softmax(x, dim=1)


# Load the dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Instantiate the model
model = GCN(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes)

# Function to freeze layers
def set_requires_grad(model, layer_index, requires_grad):
    layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5]
    for i, layer in enumerate(layers):
        for param in layer.parameters():
            param.requires_grad = requires_grad if i <= layer_index else False


# Train and evaluate the model
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


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

for layer_index in range(5):
    # Freeze all layers except the first `layer_index + 1` layers
    set_requires_grad(model, layer_index, requires_grad=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

    print(f"Training with layers up to layer {layer_index + 1} unfrozen.")

    for epoch in range(num_epochs_per_stage):
        loss = train(model, data, optimizer)
        acc = evaluate(model, data)
        print(f'Epoch: {epoch + 1 + layer_index * num_epochs_per_stage}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        # Compare performance and store the best model
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()

    # Print unfrozen layer details
    print(f"Unfreezing layer {layer_index + 2}.")

# Load the best model state
model.load_state_dict(best_model_state)
final_acc = evaluate(model, data)
print(f'Best accuracy achieved: {final_acc:.4f}')
