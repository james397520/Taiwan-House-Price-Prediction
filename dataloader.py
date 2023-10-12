import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
# import zipcodetw
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pickle



# Z-Score Normalization Function
def z_score_normalize(df, columns, save_scaler_path=None):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    
    if save_scaler_path:
        with open(save_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
    return df

# Min-Max Normalization Function
def min_max_normalize(df, columns, save_scaler_path=None):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    
    if save_scaler_path:
        with open(save_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
    return df

# Denormalization Function for Z-Score
def z_score_denormalize(df, columns, scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    df[columns] = scaler.inverse_transform(df[columns])
    return df

# Denormalization Function for Min-Max
def min_max_denormalize(df, columns, scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    df[columns] = scaler.inverse_transform(df[columns])
    return df



# Custom Dataset Class with Normalization Option
class HousePriceTrainDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_column, normalize_columns=None):
        self.dataframe = dataframe.copy()  # Creating a copy to avoid modifying the original dataframe
        
        # Applying the specified normalization methods to the specified columns
        if normalize_columns:
            for column, method in normalize_columns.items():
                if method == 'z-score':
                    self.dataframe = z_score_normalize(self.dataframe, [column],save_scaler_path ="pkl/" + column + "_z_score_normalize_data.pkl")
                elif method == 'min-max':
                    self.dataframe = min_max_normalize(self.dataframe, [column],save_scaler_path ="pkl/" + column + "_min_max_normalize_data.pkl")
        self.dataframe = min_max_normalize(self.dataframe, [target_column],save_scaler_path="pkl/" + target_column + "_min_max_normalize_data.pkl")

        self.features = self.dataframe[feature_columns].values
        self.target = self.dataframe[target_column].values
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = {'features': torch.tensor(self.features[idx], dtype=torch.float32), 
                  'target': torch.tensor(self.target[idx], dtype=torch.float32)}
        return sample


class HousePriceTestDataset(Dataset):
    def __init__(self, dataframe, feature_columns, normalize_columns=None):
        self.dataframe = dataframe.copy()  # Creating a copy to avoid modifying the original dataframe
        
        # Applying the specified normalization methods to the specified columns
        if normalize_columns:
            for column, method in normalize_columns.items():
                if method == 'z-score':
                    self.dataframe = z_score_normalize(self.dataframe, [column],save_scaler_path ="pkl/" + column + "_z_score_normalize_data.pkl")
                elif method == 'min-max':
                    self.dataframe = min_max_normalize(self.dataframe, [column],save_scaler_path ="pkl/" + column + "_min_max_normalize_data.pkl")

        
        self.features = self.dataframe[feature_columns].values


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = {'features': torch.tensor(self.features[idx], dtype=torch.float32)
                  }
        return sample


if __name__ == "__main__":
    # Load dataset
    # train_dataset = HousePriceDataset('data/training_data.csv')
    # print(train_dataset)


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
    normalized_dataset = HousePriceTrainDataset(data, selected_features, target_column, normalize_columns)

    # 訪問標準化後的數據集中的樣本
    sample = normalized_dataset[0]  # 這將顯示標準化後的第一個樣本
    print(sample)