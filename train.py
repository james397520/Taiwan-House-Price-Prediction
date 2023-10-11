import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import HousePriceModel
from data import HousePriceDataset



# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 100

data = pd.read_csv('data/training_data.csv')
# 指定要標準化的列和標準化方法
normalize_columns = {
'橫坐標': 'min-max', #z-score
'縱坐標': 'min-max'
}

# 選擇要用作特徵的列
selected_features = ['橫坐標', '縱坐標']
target_column = '單價'

# 創建標準化後的數據集
train_dataset = HousePriceDataset(data, selected_features, target_column, normalize_columns, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# print(train_dataset[0]["features"].shape)
# Initialize model
model = HousePriceModel(train_dataset[0]["features"].shape[0])#.cuda()

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        data = batch['features']
        targets = batch['target']
    # for batch_idx, (data, targets) in enumerate(train_loader):
    #     data, targets = data, targets #cuda()
    #     print(data)
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