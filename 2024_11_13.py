# %%
# 라이브러리 불러오기
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

#%%
# 로또 데이터셋 클래스 선언
class LottoDataset(Dataset):
    def __init__(
        self,
        x_samples,
        y_samples,
        idx_range
    ) -> None:
        self.x_samples = x_samples[idx_range[0]:idx_range[1]]
        self.y_samples = y_samples[idx_range[0]:idx_range[1]]
        self.idx_range = idx_range

    def __len__(self):
        return len(self.x_samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_samples[idx], dtype=torch.float32)
        y = torch.tensor(self.y_samples[idx], dtype=torch.float32)
        return x, y

# %%
# 원 핫 인코딩 함수 선언
def one_hot_encoding(numbers):
    ohbin = np.zeros(45)

    for num in numbers:
        ohbin[int(num) - 1] = 1
    return ohbin

def ohbin_to_numbers(ohbin):
    numbers = [i + 1 for i in range(len(ohbin)) if ohbin[i] == 1]
    return numbers

# %%
# 원 핫 인코딩 테스트
numbers = [3, 41, 17, 2, 9]

print(one_hot_encoding(numbers))
print(ohbin_to_numbers(one_hot_encoding(numbers)))

# %%
# 데이터 불러오기
lotto_1 = pd.read_csv('dataset/lotto_1.csv')
lotto_2 = pd.read_csv('dataset/lotto_2.csv')

# %%
# 데이터 합치기
lotto = pd.concat([lotto_1, lotto_2], axis=0)
lotto_numbers = lotto[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb']].values

# %%
# 원 핫 인코딩
ohbins = list(map(one_hot_encoding, lotto_numbers))
ohbins

# %%
# 데이터셋 생성
seq_length = 5

x_samples = [ohbins[i:(i + seq_length)] for i in range(len(ohbins) - seq_length)]
y_samples = [ohbins[i] for i in range(seq_length, len(ohbins))]
idx_range = [0, len(x_samples)]
print(x_samples[0])
print(y_samples[0])

# %%
# 데이터셋 길이 선언
total_sample = len(x_samples)
train_idx = (0, int(total_sample*0.8))
valid_idx = (int(total_sample*0.9), int(total_sample))
test_idx = (int(total_sample*0.8), int(total_sample*0.9))

# %%
# 로또 데이터셋 선언
train_dataset = LottoDataset(x_samples, y_samples, train_idx)
valid_dataset = LottoDataset(x_samples, y_samples, valid_idx)
test_dataset = LottoDataset(x_samples, y_samples, test_idx)

print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))

# %%
# 로또 데이터로더 선언
dataloader_config = {
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 0,
    'pin_memory': True
}

train_loader = DataLoader(train_dataset, **dataloader_config)
valid_loader = DataLoader(valid_dataset,**dataloader_config)
test_loader = DataLoader(test_dataset,**dataloader_config)

# %%
# 모델 선언
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LottoPredictor(nn.Module):
    def __init__(self):
        super(LottoPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=45, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 45)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, 128),
            torch.zeros(1, batch_size, 128)
        )

# %%
# 모듈 선언
model = LottoPredictor().to(device)
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 1000

# %%
# 학습
torch.cuda.empty_cache()

for epoch in range(epochs):
    # train
    model.train()
    train_loss = 0.0

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss}')

    # valid
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(valid_loader):
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            loss = criterion(output, y)
            valid_loss += loss.item()

    valid_loss /= len(valid_loader)
    print(f'Epoch: {epoch + 1}/{epochs}, Valid Loss: {valid_loss}')


# %%
# 모델 테스트
model.eval()
test_loss = 0.0

with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        output, _ = model(x)
        loss = criterion(output, y)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Epoch: {epoch + 1}/{epochs}, Test Loss: {test_loss}')

# %%
# 데이터셋 테스트
x, y = next(iter(test_loader))
output, _ = model(x.to(device))
print(x[0])
print(y[0])
print(output[0])


# %%
# 예측 결과 확인
output = ohbin_to_numbers(output[0])
print(output)
# %%
