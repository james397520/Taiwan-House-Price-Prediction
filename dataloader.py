import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import os
# import zipcodetw
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

def check_and_create_directory(directory_name):
    # 組合完整的資料夾路徑
    dir_path = os.path.join(os.getcwd(), directory_name)
    
    # 檢查資料夾是否存在
    if not os.path.exists(dir_path):
        # 如果不存在，則建立資料夾
        os.makedirs(dir_path)
        print(f"Directory '{directory_name}' created.")
    else:
        print(f"Directory '{directory_name}' already exists.")


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

# 定義 sigmoid 函數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

# 加入 One-Hot Encoding 的功能
def one_hot_encode(ids, num_classes):
    ids = torch.tensor(int(ids))
        
    # 使用 PyTorch 的 one_hot 函數來轉換
    one_hot_encoded = F.one_hot(ids, num_classes)
    return one_hot_encoded.float()


# Custom Dataset Class with Normalization Option
class HousePriceTrainDataset(Dataset):
    def __init__(self, dataframe, target_column, normalize_columns=None):
        # Load area data
        self.dataframe = dataframe.copy()  # Creating a copy to avoid modifying the original dataframe

        feature_list=[]
        check_and_create_directory("pkl")
        # Applying the specified normalization methods to the specified columns
        if normalize_columns:
            for column, method in normalize_columns.items():
                if method == 'z-score':
                    self.dataframe = z_score_normalize(self.dataframe, [column],save_scaler_path ="pkl/" + column + "_z_score_normalize_data.pkl")
                elif method == 'min-max':
                    self.dataframe = min_max_normalize(self.dataframe, [column],save_scaler_path ="pkl/" + column + "_min_max_normalize_data.pkl")
                feature_list.append(column)

        print('feature_list',feature_list)

        self.dataframe = min_max_normalize(self.dataframe, [target_column],save_scaler_path="pkl/" + target_column + "_min_max_normalize_data.pkl")
        # self.dataframe = z_score_normalize(self.dataframe, [target_column],save_scaler_path="pkl/" + target_column + "_z_score_normalize_data.pkl")

        self.features = self.dataframe[feature_list].values
        self.target = self.dataframe[target_column].values
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # print(self.features[idx])
        input = self.features[idx]
        area_code = one_hot_encode(input[0],368)
        # print('area_code',area_code.shape)
        sample = {'features': [area_code,torch.tensor(input[1:], dtype=torch.float32)], 
                  'target': torch.tensor(self.target[idx], dtype=torch.float32)}
        return sample


class HousePriceTestDataset(Dataset):
    def __init__(self, dataframe, normalize_columns=None):
        self.dataframe = dataframe.copy()  # Creating a copy to avoid modifying the original dataframe
        feature_list = []
        # Applying the specified normalization methods to the specified columns
        if normalize_columns:
            
            for column, method in normalize_columns.items():
                if method == 'z-score':
                    self.dataframe = z_score_normalize(self.dataframe, [column],save_scaler_path ="pkl/" + column + "_z_score_normalize_data.pkl")
                elif method == 'min-max':
                    self.dataframe = min_max_normalize(self.dataframe, [column],save_scaler_path ="pkl/" + column + "_min_max_normalize_data.pkl")
                feature_list.append(column)

        
        self.features = self.dataframe[feature_list].values


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input = self.features[idx]
        area_code = one_hot_encode(input[0],368)
        # print('area_code',area_code.shape)
        sample = {'features': [area_code,torch.tensor(input[1:], dtype=torch.float32)]
                  }
        return sample


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('data/reordered_final_training_data.csv')
    # 指定要標準化的列和標準化方法
    normalize_columns = {
    '行政區編號': 'one-hot-code',
    '橫坐標': 'min-max', #z-score
    '縱坐標': 'min-max'
    }

    # 選擇要用作特徵的列
    selected_features = ['橫坐標', '縱坐標']
    target_column = '單價'

    # 創建標準化後的數據集
    # normalized_dataset = HousePriceTrainDataset(data, selected_features, target_column, normalize_columns)
    train_dataset = HousePriceTrainDataset(data, target_column, normalize_columns)

    # 訪問標準化後的數據集中的樣本
    sample = train_dataset[0]  # 這將顯示標準化後的第一個樣本
    print(sample)