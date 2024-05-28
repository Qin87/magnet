import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

def set_requires_grad(model, layer_index, requires_grad):
    for i, conv in enumerate(model.convs):
        for param in conv.parameters():
            param.requires_grad = requires_grad if i <= layer_index else False
# Define the GCN model with 5 layers
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=8):
        super(GCN, self).__init__()
        self.num_layers = num_layers
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

# Instantiate the model
model = GCN(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes, num_layers=8)

# Train the model
def train(model, data, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out_list = model(data.x, data.edge_index)
        loss = sum(F.nll_loss(out[data.train_mask], data.y[data.train_mask]) for out in out_list)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            acc_list = evaluate(model, data)
            acc_str = ", ".join([f"{acc:.4f}" for acc in acc_list])
            # print(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, Accuracy: [{acc_str}]')

# Evaluate the model
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

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop with layer freezing/unfreezing
num_epochs_per_stage = 1000
best_acc = 0
best_model_state = None
best_layer_index = -1
not_improve=0
for layer_index in range(1, 8):  # From layer 1 to layer 5
    # Unfreeze up to the current layer
    set_requires_grad(model, layer_index, requires_grad=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)

    print(f"Training with layers up to layer {layer_index} unfrozen.")
    not_improve = 0
    for epoch in range(num_epochs_per_stage):
        train(model, data, optimizer, num_epochs=10)
        acc_list = evaluate(model, data)
        acc_str = ", ".join([f"{acc:.4f}" for acc in acc_list])

        # Compare performance and store the best model
        if acc_list[-1] > best_acc:
            best_acc = acc_list[-1]
            best_model_state = model.state_dict()
            best_layer_index = layer_index
        else:
            not_improve += 1
        if not_improve> 80:
            print('Not improve for 80 epochs')
            break

    # Evaluate the model on test data
    acc_list = evaluate(model, data)
    for i, acc in enumerate(acc_list):
        print(f'Accuracy of layer {i+1}: {acc:.4f}')
