from torch import nn , optim, from_numpy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests

# Preproccessing Data for a specific time horizon K
K = 24
df = pd.read_csv("parking_1h.csv")
p1 = df[df["carpark_id"] == 51].sort_values("id")
X_original = p1["current_carpark_full_total"].to_numpy()
X_original = X_original[X_original <= 87]

X_std, X_mean = np.std(X_original), np.mean(X_original)
X_norm = (X_original - X_mean) / X_std

Y = []
X = []
for i in range(len(X_norm) - K):
  X.append(X_norm[i: K + i])
  Y.append(X_norm[K + i])

X , Y = np.array(X), np.array(Y).reshape(-1, 1)
mix = np.hstack((X, Y))
np.random.shuffle(mix)
X, Y = mix[:, :K], mix[:, K]
train_ratio = 0.9
split_index = int(train_ratio * len(X))
X_train, Y_train = X[:split_index], Y[:split_index]
X_test, Y_test = X[split_index:], Y[split_index:]

X_train_tensor = from_numpy(X_train).float()
Y_train_tensor = from_numpy(Y_train).float()
X_test_tensor = from_numpy(X_test).float()
Y_test_tensor = from_numpy(Y_test).float()

class ModelBase(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.l2 = nn.Linear(K, 256)
    self.l4 = nn.Linear(256, 128)
    self.l6 = nn.Linear(128, 64)
    self.l7 = nn.Linear(64,1)

  def forward(self, x):
    x = nn.functional.sigmoid(self.l2(x))
    x = nn.functional.tanh(self.l4(x))
    x = nn.functional.tanh(self.l6(x))
    x = self.l7(x)
    return x
#  Training process
epochs = 200
lr = 1e-4
batch_size = 1
model = ModelBase()
opt = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
batches_per_epoch = len(X_train_tensor) // batch_size

# Turn on interactive mode
plt.ion()

# Create a figure and axis for loss and prediction plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
losses = []

for i in range(epochs):
    loss_total = 0
    for b in range(batches_per_epoch):
        k = b * batch_size
        x, y = X_train_tensor[k:k + batch_size], Y_train_tensor[k:k + batch_size]
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss_total += loss.item()
        loss.backward()
        opt.step()
    avg_loss = loss_total / batches_per_epoch
    losses.append(avg_loss)

    if i % 1 == 0 or i == epochs - 1:
        print(f"Epoch {i} Loss: {avg_loss:.4f}")
        with torch.no_grad():
            _pred = model(X_test_tensor[:96]).numpy()
            ax2.clear()  # Clear the previous prediction plot
            ax2.plot(_pred, label="Pred")
            ax2.plot(Y_test_tensor[:96], label="Actual")
            ax2.legend()
            ax2.set_title(f"Epoch {i + 1}")

            ax1.clear()  # Clear the previous loss plot
            ax1.plot(losses, marker='o', color='b', label='Training Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss Curve')

            plt.pause(0.1)

# Turn off interactive mode after the loop finishes
plt.ioff()

# Finalize the plots
plt.figure(figsize=(10, 5))
plt.plot(losses, marker='o', color='b', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()