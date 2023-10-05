import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import HousePriceModel

# Custom dataset
class HousePriceDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values

        # Standardize features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 100

# Load dataset
train_dataset = HousePriceDataset('data/training_data_processed.csv')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = HousePriceModel(train_dataset.X.shape[1]).cuda()

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.cuda(), targets.cuda()

        # Forward pass
        scores = model(data)
        loss = criterion(scores.squeeze(1), targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')