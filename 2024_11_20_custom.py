# %%
# 라이브러리 불러오기
# 데이터셋 및 시각화
import torch
import matplotlib.pyplot as plt

# 데이터셋 및 데이터로더
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

# 모델 선언 및 학습
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# %%
# cos 데이터셋 생성
x = torch.linspace(0, 2*torch.pi, 1000).unsqueeze(1)
y = torch.cos(x)
x_, y_ = x, y

# %%
# 데이터셋 시각화
plt.figure()
plt.plot(x, y)
plt.show()

# %%
# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx) -> Tensor:
        return self.x[idx], self.y[idx]
    
# %%
# 데이터셋 저장
dataset = CustomDataset(x, y)
torch.save(dataset, 'dataset/cos_dataset.pth')

# %%
# 데이터셋 불러오기
saved_dataset = torch.load('dataset/cos_dataset.pth')
print(saved_dataset)

# %%
# 모델 클래스 정의
class Cos(nn.Module):
    def __init__(self):
        super(Cos, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x)

# %%
# 토치 모듈 선언
model = Cos()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataloader = DataLoader(saved_dataset, batch_size=64, shuffle=True)
tensorboard = SummaryWriter(log_dir='runs/cos')

# %%
# 모델 학습
epoch = 1000

for e in range(epoch):
    for x, y in dataloader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    if e % 100 == 0:
        print(f'epoch: {e}, loss: {loss.item()}')

# %%
# 테스트
model.eval()
with torch.no_grad():
    x_test = torch.linspace(0, 2*torch.pi, 1000).unsqueeze(1)
    y_test = model(x_test)

# %%
# 시각화
plt.figure()
plt.plot(x_, y_, label='True cos values')
plt.plot(x_test, y_test, label='Model predictions')
plt.legend()
plt.show()

# %%
