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
