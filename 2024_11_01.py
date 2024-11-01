# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# %%
# scikit-learn 설치 (ipykernel)
# %pip install scikit-learn
# %%
data_1 = pd.read_csv('dataset/lotto_1.csv') # 1 ~ 600
data_2 = pd.read_csv('dataset/lotto_2.csv') # 601 ~ 

data = pd.concat([data_1, data_2], axis=0)
print(data.info())
# %%
price_columns = [
    'win1_pric',
    'win2_pric',
    'win3_pric',
    'win4_pric',
    'win5_pric'
]
# '원' 제거
for col in price_columns:
    data[col] = data[col].str.replace('원', '').str.replace(',', '').astype(np.int64)
    print(data[col])
# %%
plt.figure(figsize=(10, 6))
data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb']].hist(bins=50)
plt.show()
# %%
columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
recommandations = {}

for col in columns:
    most_common_number = data[col].value_counts().idxmax()
    recommandations[col] = most_common_number

print(recommandations)
# %%
def weighted_random_choice(columns):
    value_count = data[columns].value_counts()
    numbers = value_count.tolist()
    weights = value_count.values.tolist()

    rand = random.choices(numbers, weights=weights, k=1)
    return rand[0]
# %%
for col in columns:
    recommandations[col] = weighted_random_choice(col)

print('추천번호 : ', recommandations)
# %%
data.set_index('iso', inplace=True)
data.sort_index(inplace=True)
data.head()
# %%
lotto_numbers = data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb']]
print(lotto_numbers.info())
# %%
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(lotto_numbers)

print(scaled_data)
print(scaled_data.shape)
# %%
def create_sequence(data, seq_len):
    sequence = []
    target = []
    for i in range(len(data) - seq_len):
        sequence.append(data[i:i+seq_len])
        target.append(data[i+seq_len])
    return np.array(sequence), np.array(target)
# %%
sequence_length = 10
X, y = create_sequence(scaled_data, sequence_length)

print(X.shape, y.shape)
# %%
import torch
import torch.nn as nn

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LottoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LottoLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
# %%
model = LottoLSTM(input_size=7, hidden_size=64, num_layers=2, output_size=7).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# %%
from tqdm import tqdm

num_epochs = 100
for epoch in tqdm(range(num_epochs)):
    outputs = model(x_train.to(device))
    loss = criterion(outputs, y_train.to(device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
# %%
model.eval()
with torch.no_grad():
    test_result = model(x_test.to(device))
    loss = criterion(test_result, y_test.to(device))
    print(f'Test Loss: {loss.item()}')
    print(test_result, y_test)

# %%
# test_result로부터 추천된 번호 출력
x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
last_sequence = x_tensor[-1].unsqueeze(0)
with torch.no_grad():
    pred = model(last_sequence)

pred = scaler.inverse_transform(pred.detach().cpu().numpy())
print(pred)
# %%
