# %%
# 라이브러리 불러오기
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
# %%
# 데이터 불러오기
lotto_1 = pd.read_csv('dataset/lotto_1.csv')
lotto_2 = pd.read_csv('dataset/lotto_2.csv')
lotto = pd.concat([lotto_1, lotto_2], axis=0)
lotto.info()

# %%
# EDA
numbers = lotto[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb']]
numbers.info()
wins = lotto[['win1', 'win2', 'win3', 'win4', 'win5']]
wins.info()

# %%
# numbers and wins heatmap
heatmap_data = pd.concat([numbers, wins], axis=1)
sns.heatmap(heatmap_data.corr(), annot=True)
plt.show()

# %%
# train, val, test 데이터셋 분리
train = 0.8
val = 0.1

train_size = int(train*len(numbers))
val_size = int(val*len(numbers))
test_size = len(numbers) - train_size - val_size

train_data, val_data, test_data = random_split(numbers, [train_size, val_size, test_size])

# %%
# min-max 정규화 클래스
class MinMaxScaler():
    def __init__(self) -> None:
        self.min = None
        self.max = None

    def fit_transform(self, data: pd.DataFrame) -> float:
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return (data - self.min)/(self.max - self.min)
    
    def inverse_transform(self, data) -> float:
        return data*(self.max - self.min) + self.min

# %%
# 커스텀 데이터셋 (LSTM)
class LottoDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            seq_len: int,
            scaler: MinMaxScaler
    ) -> None:
        self.data = data
        self.seq_len = seq_len
        self.scaler = scaler
        self.scaled_data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

    def __len__(self):
        return len(self.scaled_data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.scaled_data.iloc[idx:idx + self.seq_len]
        y = self.scaled_data.iloc[idx + self.seq_len]

        x = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        return x, y

# %%
# 데이터 로더
class LottoDataLoader(DataLoader):
    def __init__(
            self,
            data: Dataset = LottoDataset,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 8,
            pin_memory: bool = True
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

# %%
# LSTM 모델
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LottoLSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout: float = 0.0,
            init_weight: bool = True
    ) -> None:
        super(LottoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        # 가중치 초기화
        if init_weight:
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# %%
# 학습 클래스 (train, val 데이터셋 활용)
class LottoTrainer():
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            train_data: DataLoader,
            val_data: DataLoader,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_data = train_data
        self.val_data = val_data

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            self.model.train()
            for x, y in tqdm(self.train_data):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    def test(self) -> None:
        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_data:
                x, y = x.to(self.device), y.to(self.device)
                test_result = self.model(x)
                loss = self.criterion(test_result, y)
                print(f'Test Loss: {loss.item()}')
                print(test_result, y)
        return test_result

# %%
# 모듈 선언
dataset = LottoDataset(data=numbers, seq_len=10, scaler=MinMaxScaler())
dataloader = LottoDataLoader(data=dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
dataloader = dataloader.get_loader()

model = LottoLSTM(input_size=7, hidden_size=128, num_layers=8, output_size=7).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

trainer = LottoTrainer(model=model, criterion=criterion, optimizer=optimizer, device=device, train_data=dataloader, val_data=dataloader)

# %%
# 학습
trainer.train(num_epochs=10)

# %%
# 테스트
test_result = trainer.test()
    
# %%
# test_result 역정규화
test_result = pd.DataFrame(test_result, columns=numbers.columns)
test_result = dataset.scaler.inverse_transform(test_result)

# 정수로 변환
test_result = test_result.astype(int)
test_result
# %%
