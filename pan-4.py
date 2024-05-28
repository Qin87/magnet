import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

from nets.DiG_NoConv import DiG_SimpleXBN_nhid_Pan


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.num_layers = 8
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

    def forward(self, x, edge_index, w_layer):
        out_list = []
        for i in range(self.num_layers):
            if w_layer[i] == 1:
                x = self.convs[i](x, edge_index)
                if i < self.num_layers - 1:
                    x = F.relu(x)
                out_list.append(F.log_softmax(x, dim=1))
                return out_list[0]
        output = sum(out_list)
        return output

def set_requires_grad(model, w_layer):
    # for i, layer in enumerate(model.convs):
    for i, layer in enumerate(model.convx):
        if w_layer[i] == 1:
            for param in layer.parameters():
                param.requires_grad = True
        else:
            for param in layer.parameters():
                param.requires_grad = False

def train(model, data, optimizer, w_layer, num_epochs=10):
    model.train()
    set_requires_grad(model, w_layer)
    optimizer.zero_grad()
    # out = model(data.x, data.edge_index, w_layer)
    out = model(data.x, data.edge_index, w_layer)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, w_layer):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, w_layer)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
    return acc

# Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Initialize GCN model
# model = GCN(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes)
model = DiG_SimpleXBN_nhid_Pan(input_dim=dataset.num_node_features, hid_dim=64, out_dim=dataset.num_classes, dropout=0.5, layer=8)
# Training loop with layer combinations
num_epochs_per_stage = 10
best_acc = 0.0
best_layers = None
best_model_state = None

# Loop through different layer combinations
for i in range(1, 9):  # From 1 to 8 layers
    w_layer = [1] * i + [0] * (8 - i)  # One-hot encoding for selecting i layers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"Training with {i} layers unfrozen.")

    for epoch in range(num_epochs_per_stage):
        loss = train(model, data, optimizer, w_layer)
        acc = evaluate(model, data, w_layer)
        print(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        # Track the best model
        if acc > best_acc:
            best_acc = acc
            best_layers = i
            best_model_state = model.state_dict()

# Load the best model state
model.load_state_dict(best_model_state)

# Evaluate the best model
w_layer_best = [1] * best_layers + [0] * (8 - best_layers)
final_acc = evaluate(model, data, w_layer_best)
print(f'Best accuracy achieved with {best_layers} layers:{best_acc:.4f}, final: {final_acc:.4f}')
