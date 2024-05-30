import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from args import parse_args
import argparse


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

    def forward(self, x, edge_index, w_layer):
        out_list = []
        for i in range(self.num_layers):
            if w_layer[i] == 1:
                x = self.convs[i](x, edge_index)
                if i < self.num_layers - 1:
                    x = F.relu(x)
                out_list.append(F.log_softmax(x, dim=1))
        output = sum(out_list)
        return output
        # # w_layer = args.pan
        # out_list = []
        # for i in range(self.num_layers):
        #     x = self.convs[i](x, edge_index)
        #     if i < self.num_layers - 1:
        #         x = F.relu(x)
        #     out_list.append(F.log_softmax(x, dim=1))
        # output= sum(weight * value for weight, value in zip(w_layer, out_list))
        # return output

# Load the dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Instantiate the model
model = GCN(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes, num_layers=8)
args = parse_args()
args.pan = 8* args.pan
print(args.pan)

# Train the model
# def train(model, data, optimizer, num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         out_list = model(data.x, data.edge_index, args.pan)
#         loss = sum(F.nll_loss(out[data.train_mask], data.y[data.train_mask]) for out in out_list)
#         loss.backward()
#         optimizer.step()
#         if (epoch + 1) % 10 == 0:
#             acc_list = evaluate(model, data)
#             acc_str = ", ".join([f"{acc:.4f}" for acc in acc_list])
#             # print(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, Accuracy: [{acc_str}]')

def train(model, data, optimizer, w_layer, num_epochs=10):
    model.train()
    set_requires_grad(model, 0, requires_grad=False)
    set_requires_grad(model, 1, requires_grad=False)
    set_requires_grad(model, 2, requires_grad=False)
    set_requires_grad(model, 3, requires_grad=False)
    set_requires_grad(model, 4, requires_grad=False)
    set_requires_grad(model, 5, requires_grad=False)
    set_requires_grad(model, 6, requires_grad=False)
    set_requires_grad(model, 7, requires_grad=False)

    optimizer.zero_grad()
    out = model(data.x, data.edge_index, w_layer)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

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

# Training loop with layer freezing/unfreezing
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
print(f'Best accuracy achieved with {best_layers} layers: {final_acc:.4f}')
