import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader

project_root = str(Path(__file__).parent.parent)
data = [[] for _ in range(10)]
training_datasets = []
training_dataloaders = []
testing_datasets = []
testing_dataloaders = []
batch_size = 64
for i in range(10):
    veh_data_dir = project_root+ f"/data/veh_{i}.npy"
    data[i] = np.load(veh_data_dir)
    train_index = round(0.85 * len(data[i]))
    train_tensor = torch.FloatTensor(data[i][:train_index])
    test_tensor = torch.FloatTensor(data[i][train_index:])
    train_dataset = TensorDataset(train_tensor)
    test_dataset = TensorDataset(test_tensor)
    training_datasets.append(train_dataset)
    testing_datasets.append(test_dataset)
    training_dataloaders.append(DataLoader(train_dataset, batch_size=batch_size))
    testing_dataloaders.append(DataLoader(test_dataset, batch_size=batch_size))

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        # layers = [nn.Linear(input_size, layer_sizes[0])]
        # layers += [nn.ReLU(True)]
        # for i in range(1, len(layer_sizes[1:])):
        #     layers += [nn.Linear(layer_sizes[i-1], layer_sizes[i])] 
        #     layers += [nn.ReLU(True)]
        # layers += [nn.Linear(layer_sizes[-1], output_size)]
        # self.model = nn.Sequential(*layers)

        self.linera_relu_stack = nn.Sequential(
            nn.Linear(22, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linera_relu_stack(x)
        return logits

# model = Net().to(device)
# print(model)
model = Net()

loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, tensor in enumerate(dataloader):
        x = tensor[0][:, [1,2] + list(range(4, tensor[0].size(dim=1)))]
        y = tensor[0][:, 3:4]
        y = y.squeeze(dim=1)
        y = y.long()
        print(y.shape)

        # x, y = x.to(device), y.to(device)

        pred = model(x)
        pred = pred.float()
        print('here')
        loss = loss_fn(pred, y)
        print('here1')

        loss.backward()
        print('here2')
        optimizer.step()
        print('here3')
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    for i, tensor in enumerate(dataloader):
        # send the input/labels to the GPU
        x = tensor[0][:, 1:3]
        y = tensor[0][:, 3:]
        y = y.long()

        # x, y = x.to(device), y.to(device)

        # forward
        with torch.set_grad_enabled(False):
            pred = model(x)
            pred = pred.long()
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
    # test(test_dataloader, model, loss_fn)
print("Done!")