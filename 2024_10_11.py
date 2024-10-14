# %%
import pandas as pd
import numpy as np
# %%
data = pd.read_csv('dataset/survey_results_public.csv')
data.info()
data.head()
# %%
# 데이터 분석 (다음주 다시)
# %%
# 파이토치 실습
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
# %%
x = torch.linspace(0, 2*math.pi, 1000).unsqueeze(1)
y = torch.sin(x)

print(x)
print(y)

plt.figure()
plt.plot(x, y)
plt.show()
# %%
model = nn.Sequential(
    nn.Linear(1, 100),
    nn.LeakyReLU(inplace=True),
    nn.Linear(100, 80),
    nn.LeakyReLU(inplace=True),
    nn.Linear(80, 60),
    nn.LeakyReLU(inplace=True),
    nn.Linear(60, 40),
    nn.LeakyReLU(inplace=True),
    nn.Linear(40, 20),
    nn.LeakyReLU(inplace=True),
    nn.Linear(20, 1)
)
lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
# %%
model.eval()
with torch.no_grad():
    y_pred = model(x)
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, y_pred)
    plt.show()

# %%
