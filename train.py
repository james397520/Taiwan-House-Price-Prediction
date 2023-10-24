import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from model.model import HousePriceModel, TransformerRegressor
from model.model_v3 import HousePriceModel, TransformerRegressor
from model.model_v2 import HousePriceModel_CNN
from dataloader import HousePriceTrainDataset
import platform

def main():

    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-3
    epochs = 100

    data = pd.read_csv('data/reordered_final_training_data.csv')
    # 指定要標準化的列和標準化方法
    normalize_columns = {
    # '地區':'one-hot-encoding',
    # '使用分區':'one-hot-encoding',
    # '主要用途':'one-hot-encoding',
    # '主要建材':'one-hot-encoding',
    # '建物型態':'one-hot-encoding',
    '行政區編號': 'one-hot-code',
    '土地面積': 'min-max',
    '移轉層次': 'min-max',
    '總樓層數': 'min-max',
    '屋齡': 'min-max',
    '建物面積': 'min-max',
    '車位面積': 'min-max',
    '車位個數': 'min-max',
    '橫坐標': 'min-max', #z-score
    '縱坐標': 'min-max',
    '主建物面積': 'min-max',
    '陽台面積': 'min-max',
    '附屬建物面積': 'min-max',
    }

    target_column = '單價'
    if platform.system() == "Linux":
        gpu = True
    elif platform.system() == "Darwin":
        gpu = False

    # 創建標準化後的數據集
    train_dataset = HousePriceTrainDataset(data, target_column, normalize_columns)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(normalize_columns.keys())-1
    print(input_dim)
    # Initialize model
    model = HousePriceModel(input_dim)
    # model = TransformerRegressor(input_dim, 4, 6)
    # model = HousePriceModel_CNN(input_dim)

    if gpu:
        model = model.cuda()
    else:
        model = model

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for batch in train_loader:
            # print(batch['features'].shape)
            # print(batch['target'].shape)
            if gpu:
                data = batch['features']
                data[0] = data[0].cuda()
                data[1] = data[1].cuda()
                targets = batch['target'].cuda()
            else:
                data = batch['features']
                targets = batch['target']

            scores = model(data)
            loss = criterion(scores.squeeze(1), targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'model_cnn.pth')



if __name__ == "__main__":
    main()