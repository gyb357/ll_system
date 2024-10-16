# %%
# 1. survey_results_public.csv 파일을 읽어서 연령대 별로 응답자수를 bar그래프로 표시
# (reindex 하여 보기좋게 순서대로 배치.)

# 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('dataset/survey_results_public.csv')
# print(df.head())

# 데이터 전처리
reversed_df = df[
    [
        'Age'
    ]
]
# print(reversed_df.head())
# print(reversed_df['Age'].duplicated().head())

# 연령대 별 응답자수
size_by_age = reversed_df.groupby('Age').size()
print(size_by_age.head())

# reindex
reindexed_age = size_by_age.reindex(
    index=(
        'Under 18 years old',
        '18-24 years old',
        '25-34 years old',
        '35-44 years old',
        '45-54 years old',
        '55-64 years old',
        '65 years or older'
    )
)
print(reindexed_age.head())

# 그래프 그리기
reindexed_age.plot.bar()
plt.xlabel('Age')
plt.ylabel('Number of Respondents')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
# %%
# 2. owid-covid-data.csv 데이터를 읽어 들여 한국과 프랑스의 확진자 추이를 선그래프로 표시.
# (인구 비율을 반드시 적용시킬것)

# 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('dataset/owid-covid-data.csv')

# 한국, 프랑스 데이터만 추출
kor = data[data['location'] == 'South Korea']
fra = data[data['location'] == 'France']

# 인덱스 설정
kor = kor.set_index('date')
fra = fra.set_index('date')

# 인구 비율 계산
kor_pop = kor['population']['2022-01-01']
fra_pop = fra['population']['2022-01-01']
rate = round(fra_pop/kor_pop, 2)

# 그래프 그리기
plt.figure()
final_data = pd.DataFrame({
    'kor': kor['total_cases']*rate,
    'fra': fra['total_cases']
})
final_data.plot(rot=45)
plt.show()
# %%
# 라이브러리 불러오기
from typing import Optional, Callable, List
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from tqdm import tqdm


# 활성화 함수 메서드
def activation_layer(activation: Optional[Callable[..., nn.Module]] = None) -> nn.Module:
    return activation if activation is not None else nn.ReLU()


# 이중 선형 블록 클래스
class DoubleLinearBlock(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        super(DoubleLinearBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias),
            activation_layer(activation),
            nn.Linear(out_features, out_features, bias=bias),
            activation_layer(activation),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


# 모델 클래스
class Model(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            hidden_depth: int,
            output_size: int,
            bias: bool = True,
            activation: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0,
            init_weights: bool = True
    ) -> None:
        super(Model, self).__init__()

        self.layers = nn.ModuleList()
        # 입력 레이어
        self.layers.append(nn.Linear(in_features, hidden_features, bias=bias))
        self.layers.append(activation_layer(activation))

        # 은닉 레이어
        for _ in range(hidden_depth):
            self.layers.append(DoubleLinearBlock(hidden_features, hidden_features, bias, activation, dropout))

        # 출력 레이어
        self.layers.append(nn.Linear(hidden_features, output_size, bias=bias))

        # 가중치 초기화 (activation 함수가 ReLU 또는 LeakyReLU)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# 학습 클래스
class Trainer():
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: optim,
            sigmoid: bool = False
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.sigmoid = sigmoid
        self.loss = []

    def train(self, x: Tensor, y: Tensor, epochs: int = 1000) -> List[float]:
        self.model.train()

        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()       # 기울기 초기화
            output = self.model(x)           # 예측

            if self.sigmoid is True:         # 시그모이드 함수 적용
                output = torch.sigmoid(output)

            loss = self.criterion(output, y) # 손실 계산
            loss.backward()                  # 역전파
            self.optimizer.step()            # 가중치 업데이트

            # 손실 기록
            if epoch % int(epochs*0.1) == 0:
                self.loss.append(loss.item())
        return self.loss

    def eval(self, x: Tensor) -> Tensor:
        self.model.eval()

        with torch.no_grad():
            output = self.model(x)

            if self.sigmoid is True:
                output = torch.sigmoid(output)
        return output


# %%
# 3. torch 를 사용하여 mode 을 만들어 xor 연산을 학습.
# (반드시 torch.nn을 사용할것 , loss가 0.01 이하달성시킬것)

# 라이브러리 불러오기
import torch

# 데이터 생성
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 모델 생성
model = Model(2, 50, 5, 1)

# 손실함수, 옵티마이저 생성
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
trainer = Trainer(model, criterion, optimizer, sigmoid=True)
loss = trainer.train(x, y, epochs=1000)

# 결과 확인
output = trainer.eval(x)
print(output)

for l in loss:
    print(l)

# %%
# 4. torch 를 사용하여 sin 그래프를 예측하는 모델을 작성하고 학습 시켜 결과를 그래프로 표현.

# 라이브러리 불러오기
import math
import matplotlib.pyplot as plt

# 데이터 생성
x = torch.linspace(0, 2*math.pi, 1000).unsqueeze(1)
y = torch.sin(x)

# 모델 생성
model = Model(1, 50, 5, 1, activation=nn.LeakyReLU())

# 손실함수, 옵티마이저 생성
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
trainer = Trainer(model, criterion, optimizer)
loss = trainer.train(x, y, epochs=1000)

# 결과 확인
output = trainer.eval(x)
print(output)

for l in loss:
    print(l)

# 그래프 그리기
plt.figure()
plt.plot(x, y)
plt.plot(x, output)
plt.show()
# %%
# 5. torch 를 사용하여 y=ax+b 를 가정하고 선형회귀 분석을 신경망모델로 만들어 학습시키고 결과를 그래프로 표현.

# 데이터 생성
a = 2
b = 10 + torch.rand(100, 1)
x = torch.linspace(0, 10, 100).unsqueeze(1)
y = a*x + b

# 모델 생성
model = Model(1, 50, 5, 1)

# 손실함수, 옵티마이저 생성
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
trainer = Trainer(model, criterion, optimizer)
loss = trainer.train(x, y, epochs=1000)

# 결과 확인
output = trainer.eval(x)
print(output)

for l in loss:
    print(l)

# 그래프 그리기
plt.figure()
plt.scatter(x, y)
plt.plot(x, output, color='red')
plt.show()
# %%
