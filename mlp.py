import sys
import os
print("Python Path:", sys.path)
print("Current Working Directory:", os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from args import parse_args

args = parse_args()



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)


# Function to create DataLoader
def create_dataloader(features, labels, mask, batch_size=64):
    dataset = TensorDataset(features, labels, mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for features, labels, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}')


# Function to evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels, masks in data_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

from data_model import load_dataset, count_homophilic_nodes
data_x, data_y, edges, edges_weight, num_features, data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin, IsDirectedGraph, edge_attr, data_batch = load_dataset(args)


no_in, homo_ratio_A, no_out,   homo_ratio_At, in_homophilic_nodes, out_homophilic_nodes, in_heterophilic_nodes, out_heterophilic_nodes, no_in_nodes, no_out_nodes = count_homophilic_nodes(edges, data_y)

split=0
data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin[:, split].clone(),
                                                                      data_val_maskOrigin[:, split].clone(),
                                                                      data_test_maskOrigin[:, split].clone())

all_list = [
    ("no_in_nodes", no_in_nodes),
    ("in_homophilic_nodes", in_homophilic_nodes),
    ("in_heterophilic_nodes", in_heterophilic_nodes),
    ("no_out_nodes", no_out_nodes),
    ("out_homophilic_nodes", out_homophilic_nodes),
    ("out_heterophilic_nodes", out_heterophilic_nodes),
    ("all_nodes", list(range(data_x.shape[0])))
]

for name, list in all_list:
    if len(list)==0:
        continue
    tensor_nodes = torch.tensor(list)

    data_x_filter = data_x[tensor_nodes]
    data_y_filter = data_y[tensor_nodes]
    data_train_mask_filter = data_train_mask[tensor_nodes]
    data_val_mask_filter = data_val_mask[tensor_nodes]
    data_test_mask_filter = data_test_mask[tensor_nodes]

    train_loader = create_dataloader(data_x_filter, data_y_filter, data_train_mask_filter)
    val_loader = create_dataloader(data_x_filter, data_y_filter, data_val_mask_filter)
    test_loader = create_dataloader(data_x_filter, data_y_filter, data_test_mask_filter)

    # Initialize Model, Loss Function, and Optimizer
    input_dim = data_x_filter.size(1)
    hidden_dim = 128
    output_dim = len(torch.unique(data_y_filter))

    model = MLP(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the Model
    print(f"Training MLP model for {name}...")
    train_model(model, train_loader, criterion, optimizer)

    # Evaluate the Model
    val_accuracy = evaluate_model(model, val_loader)
    test_accuracy = evaluate_model(model, test_loader)

    print(f'Category: {name}')
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    print(f'Test Accuracy: {test_accuracy:.2f}%\n')



