# %%
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

import numpy as np
# %%
# Input data
x = torch.linspace(0, 2 * math.pi, 1000).unsqueeze(1)  # Shape (1000, 1)
y = torch.sin(x)  # Shape (1000, 1)

# Define the model
class Model(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, hidden_depth: int, out_features: int) -> None:
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features))
        for _ in range(hidden_depth):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
        self.layers.append(nn.Linear(hidden_features, out_features))
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return x

# Model, criterion, and optimizer setup
model = Model(1, 50, 10, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
loss_list = []
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()

    loss_list.append(loss.item())  # Store loss for plotting
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Loss graph output
plt.plot(np.linspace(0, epochs, epochs), loss_list)  # Use loss_list for plotting all losses
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot true values vs model predictions
x_np = x.detach().cpu().numpy()  # Convert x to NumPy
y_np = y.detach().cpu().numpy()  # Convert y to NumPy
plt.plot(x_np, y_np, label='True values')

# Plot model predictions
predictions = model(x).detach().cpu().numpy()  # Detach model output and convert to NumPy
plt.plot(x_np, predictions, label='Model predictions')
plt.legend()
plt.show()

# %%
