import torch
from sklearn.preprocessing import StandardScaler

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