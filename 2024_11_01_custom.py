# %%
# 라이브러리 불러오기

# 데이터 전처리
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 딥러닝
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
from torch import Tensor

# 딥러닝 학습
import torch.optim as optim
from torch import device
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
# min-max 정규하
class MinMaxScaler():
    def __init__(self) -> None:
        self.min = None
        self.max = None

    def fit_transform(self, data: pd.DataFrame) -> float:
        self.min = np.min(data, axis = 0)
        self.max = np.max(data, axis = 0)
        return (data - self.min)/(self.max - self.min)

    def inverse_transform(self, data: pd.DataFrame) -> float:
        if self.min is None or self.max is None:
            Exception('Fit the data first')
        else: return data*(self.max - self.min) + self.min

# %%
# 데이터셋 클래스
class LottoDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame, # Must be scaled dataset
        seq_len: int,
    ) -> None:
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, index: int) -> tuple:
        x = self.data.iloc[index:index+self.seq_len]
        y = self.data.iloc[index+self.seq_len]

        x = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        return x, y

# %%
# 데이터 로더 클래스
class LottoDataLoader(DataLoader):
    def __init__(
        self,
        data: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
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
# LSTM 모델 클래스
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

        if init_weight:
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# %% 
# Transformer 모델 클래스
class LottoTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        output_size: int,
        dropout: float = 0.1
    ) -> None:
        super(LottoTransformer, self).__init__()
        # Embedding layer
        self.embedding = nn.Linear(input_size, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, hidden_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out

# %%
# metric 함수
def mean_absolute_error(out: Tensor, y: Tensor) -> float:
    return torch.mean(torch.abs(out - y)).item()

# %%
# 학습 클래스
columns = ['Epoch', 'Train_loss', 'Train_mae', 'Val_loss', 'Val_mae']

class Trainer():
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: nn.Module,
        device: device,
        train_data: DataLoader,
        val_data: DataLoader,
        test_data: DataLoader
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.tensorboard = SummaryWriter()

    def eval(self, data: DataLoader) -> tuple:
        data_len = len(data)
        eval_loss, eval_mae = 0, 0

        self.model.eval()
        with torch.no_grad():
            for x, y in data:
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)
                eval_loss += self.criterion(out, y).item()
                eval_mae += mean_absolute_error(out, y)

        return eval_loss/data_len, eval_mae/data_len, out

    def train(self, epochs: int) -> None:
        # train
        dataset_len = len(self.train_data)

        for epoch in tqdm(range(epochs)):
            self.model.train()
            train_loss, train_mae = 0, 0

            for x, y in tqdm(self.train_data):
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                out = self.model(x)
                loss = self.criterion(out, y)

                train_loss += loss.item()
                train_mae += mean_absolute_error(out, y)

                loss.backward()
                self.optimizer.step()
                
            train_loss /= dataset_len
            train_mae /= dataset_len

            if (epoch + 1) % 100 == 0:
                print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss}, Train MAE: {train_mae}')

            # validation
            val_loss, val_mae, _ = self.eval(self.val_data)
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch: {epoch + 1}/{epochs}, Val Loss: {val_loss}, Val MAE: {val_mae}')

            # tensorboard
            values = [epoch, train_loss, train_mae, val_loss, val_mae]
            for i in range(1, len(columns)):
                self.tensorboard.add_scalar(columns[i], values[i], epoch)

# %%
# 모듈 선언

# 데이터 전처리
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numbers)

# 데이터셋 분리
train = 0.8
val = 0.1

train_size = int(train*len(scaled_data))
val_size = int(val*len(scaled_data))
test_size = len(scaled_data) - train_size - val_size
train_data, val_data, test_data = random_split(scaled_data, [train_size, val_size, test_size])

# 데이터셋
train_dataset = LottoDataset(data=scaled_data, seq_len=10)
val_dataset = LottoDataset(data=scaled_data, seq_len=10)
test_dataset = LottoDataset(data=scaled_data, seq_len=10)

# 데이터 로더
train_dataloader = LottoDataLoader(data=train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True).get_loader()
val_dataloader = LottoDataLoader(data=val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True).get_loader()
test_dataloader = LottoDataLoader(data=test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True).get_loader()

# 모델, 손실함수, 옵티마이저, 학습
dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LottoLSTM(input_size=7, hidden_size=128, num_layers=8, output_size=7).to(dvc)
model = LottoTransformer(input_size=7, num_layers=4, hidden_dim=128, num_heads=4, output_size=7).to(dvc)
criterion = nn.MSELoss().to(dvc)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=dvc,
    train_data=train_dataloader,
    val_data=val_dataloader,
    test_data=test_dataloader
)

# %%
# 학습
trainer.train(epochs=1000)

# %%
# 테스트
test_loss, test_mae, test_result = trainer.eval(test_dataloader)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

test_result = pd.DataFrame(test_result, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb'])
test_result = scaler.inverse_transform(test_result)
test_result = test_result.astype(int)

# test actual과 비교
actual = pd.DataFrame(scaler.inverse_transform(numbers), columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb'])
actual = actual.iloc[-len(test_result):]
actual = actual.astype(int)

print(test_result)
print(actual)
# %%
