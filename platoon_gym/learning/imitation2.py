import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

project_root = str(Path(__file__).parent.parent)
data = [[] for _ in range(10)]
training_datasets = []
training_dataloaders = []
testing_datasets = []
testing_dataloaders = []
batch_size = 64
for i in range(10):
    veh_data_dir = project_root+ f"/data/20_history/veh_{i}.npy"
    data[i] = np.load(veh_data_dir)
    train_index = round(0.85 * len(data[i]))
    train_tensor = torch.FloatTensor(data[i][:train_index])
    test_tensor = torch.FloatTensor(data[i][train_index:])
    train_dataset = TensorDataset(train_tensor)
    test_dataset = TensorDataset(test_tensor)
    training_datasets.append(train_dataset)
    testing_datasets.append(test_dataset)
    training_dataloaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
    testing_dataloaders.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=True))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()

        self.linera_relu_stack = nn.Sequential(
            nn.Linear(42, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linera_relu_stack(x)
        return logits

model = Net()
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for tensor in dataloader:
        x = tensor[0][:, [1,2] + list(range(4, tensor[0].size(dim=1)))]
        y = tensor[0][:, 3:4]

        pred = model(x)
        pred = pred.float()
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / size
    print(f"Train Error: \n Avg loss: {avg_loss:>8f} \n")
    return avg_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    total_loss = 0

    for tensor in dataloader:
        # send the input/labels to the GPU
        x = tensor[0][:, [1,2] + list(range(4, tensor[0].size(dim=1)))]
        y = tensor[0][:, 3:4]

        # forward
        pred = model(x)
        total_loss += loss_fn(pred, y).item()

    avg_loss = total_loss / size
    print(f"Test Error: \n Avg loss: {avg_loss:>8f} \n")
    return avg_loss


train_dataloader = training_dataloaders[1]
test_dataloader = testing_dataloaders[1]

epochs = 450
train_loss = []
test_loss = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss.append(train(train_dataloader, model, loss_fn, optimizer))
    test_loss.append(test(test_dataloader, model, loss_fn))
print("Done!")

print('total avg training loss: ', sum(train_loss) / len(train_loss))
print('total avg testing loss: ', sum(test_loss) / len(test_loss))

plt.plot(train_loss)
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(test_loss)
plt.title('Testing Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()