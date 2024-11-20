# %%
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim

# %%
print('module load complete')
print(f'torch version: {torch.__version__}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

# %%
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)

# %%
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
# %%
dataset = CustomDataset(x, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# %%
for idx, (x, y) in enumerate(dataloader):
    print(f'idx: {idx}, x: {x}, y: {y}')

# %%
torch.save(dataset, 'dataset/xor_dataset.pth')

# %%
saved_dataset = torch.load('dataset/xor_dataset.pth')
print(saved_dataset)

# %%
for idx, (x, y) in enumerate(saved_dataset):
    print(f'idx: {idx}, x: {x}, y: {y}')

# %%
class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.linear1 = nn.Linear(2, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return self.sigmoid(x)
    
# %%
model = XOR().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# %%
for epoch in range(20000):
    for idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(f'epoch: {epoch}, idx: {idx}, loss: {loss.item()}')
            
# %%
x_test = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
y_test = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(x_test)
    print(y_pred)
    print(y_test)

# %%
