import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self, input_size, output_size, layer_sizes):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        layers = [nn.Linear(input_size, layer_sizes[0])]
        layers += [nn.ReLU(True)]
        for i in range(1, len(layer_sizes[1:])):
            layers += [nn.Linear(layer_sizes[i-1], layer_sizes[i])] 
            layers += [nn.ReLU(True)]
        layers += [nn.Linear(layer_sizes[-1], output_size)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits

def train(model, device):
    loss_function = nn.CrossEntropyLoss()
    datasets = load_data()
    for veh in 10:
        for epoch in range (100):
            print(f'epoch {epoch + 1}')

            train_epoch(model, device, loss_function, datasets)

def train_epoch(model, device, loss_func, datasets):
    model.train()



def load_data(self):
    data = [[] for _ in range(10)]
    datasets = []
    for i in range(10):
        veh_data_dir = get_project_root() + f"/data/veh_{i}.npy"
        data[i] = np.load(veh_data_dir)
        tensor = torch.FloatTensor(data[i])
        dataset = TensorDataset(tensor)
        datasets.append(dataset)
    return datasets

def get_project_root() -> str:
    """
    Returns:
        str: project root path
    """
    return str(Path(__file__).parent.parent)

if __name__ == "__main__":
    net = Net(100, 100, [64, 64])
    net.load_data()


# keeping here for time being
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

train_dataloader = training_dataloaders[0]
test_dataloader = testing_dataloaders[0]

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")