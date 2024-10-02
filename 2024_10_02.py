# %%
# %pip install matplotlib
# %pip install pandas
# %%
import pandas as pd
import matplotlib.pyplot as plt
# %%
data = pd.read_csv('dataset\owid-covid-data.csv')
print(data.info())
# %%
revise_data = data[['iso_code', 'location', 'date', 'total_cases', 'population']]
print(revise_data.info())
# %%
locations = revise_data['location'].unique()
print(locations)
# %%
korea_data = revise_data[revise_data['location'] == 'South Korea']
korea_data = korea_data.set_index('date')
print(korea_data)
# %%
korea_total_cases = korea_data['total_cases']
korea_total_cases.plot() 
# %%
usa_data = revise_data[revise_data['location'] == 'United States']
usa_data = usa_data.set_index('date')
print(usa_data)
# %%
usa_total_cases = usa_data['total_cases']
usa_total_cases.plot() 
# %%
final_data = pd.DataFrame({'kor': korea_total_cases, 'usa': usa_total_cases})
final_data.plot(rot=45)
print(final_data)
# %%
kor_pop = korea_data['population']['2022-01-01']
print(kor_pop)
usa_pop = usa_data['population']['2022-01-01']
print(usa_pop)
# %%
rate = round(usa_pop/kor_pop, 2)
print(rate)
# %%
final_data = pd.DataFrame({
    'kor': korea_total_cases*rate,
    'usa': usa_total_cases
})
final_data.plot(rot=45)
# %%
# %pip install torch
# %pip install tqdm
# %%
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

class XOR(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature):
        super(XOR, self).__init__()
        self.linear1 = nn.Linear(in_feature, hidden_feature)
        self.linear2 = nn.Linear(hidden_feature, out_feature)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

model = XOR(2, 2, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_list = []

for epoch in tqdm.tqdm(range(50000)):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        loss_list.append(loss.item())

for l in loss_list:
    print(l)


# %%
inp = torch.tensor([[1, 0]], dtype=torch.float32)
out = model(inp)
print(out)
# %%
