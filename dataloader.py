import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
# import zipcodetw
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np



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
def one_hot_encode(df):
    
    
    # 選擇需要進行 One-Hot Encoding 的列
    columns_to_encode = ['使用分區', '主要用途', '主要建材', '建物型態']
    
    # 使用 pandas 的 get_dummies 函數進行 One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=columns_to_encode)
    
    return df_encoded

# Function to get the district ID based on the city and district names
def get_district_id(city: str, district: str, area_df: pd.DataFrame) -> int:
    full_name = city + district
    print(full_name)
    # print(area_df['行政區名稱'])
    if area_df['行政區名稱'] == full_name:
        print(full_name)
    matching_row = area_df[area_df['行政區名稱'] == full_name]
    # print(matching_row)
    return matching_row['行政區編號'].values[0] if not matching_row.empty else None





# Custom Dataset Class with Normalization Option
class HousePriceTrainDataset(Dataset):
    def __init__(self, dataframe, target_column, normalize_columns=None):
        # Load area data
        area_file_path = 'data/area.csv'
        self.area_df = pd.read_csv(area_file_path)
        self.dataframe = dataframe.copy()  # Creating a copy to avoid modifying the original dataframe
        # 合併 '縣市' 和 '鄉鎮市區' 列
        self.dataframe['行政區名稱'] = self.dataframe['縣市'] + self.dataframe['鄉鎮市區']
        feature_list=[]
        # Applying the specified normalization methods to the specified columns
        if normalize_columns:
            for column, method in normalize_columns.items():
                if method == 'z-score':
                    self.dataframe = z_score_normalize(self.dataframe, [column],save_scaler_path ="pkl/" + column + "_z_score_normalize_data.pkl")
                elif method == 'min-max':
                    self.dataframe = min_max_normalize(self.dataframe, [column],save_scaler_path ="pkl/" + column + "_min_max_normalize_data.pkl")
                feature_list.append(column)
        print('QQQS',feature_list)
        # self.dataframe = one_hot_encode(self.dataframe)
        # self.dataframe = one_hot_encode(self.dataframe)
        # print(self.area_df['行政區名稱'])
        # Add a new column to the training data DataFrame to store the district IDs
        # self.dataframe['行政區編號'] = self.dataframe.apply(
        #     lambda row: get_district_id(row['縣市'], row['鄉鎮市區'], self.area_df), axis=1)


        self.dataframe = min_max_normalize(self.dataframe, [target_column],save_scaler_path="pkl/" + target_column + "_min_max_normalize_data.pkl")
        # self.dataframe = z_score_normalize(self.dataframe, [target_column],save_scaler_path="pkl/" + target_column + "_z_score_normalize_data.pkl")

        # self.dataframe[target_column].apply(sigmoid)
        # print(self.dataframe[feature_list].head)
        self.features = self.dataframe[feature_list].values
        self.target = self.dataframe[target_column].values
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        print(self.features[idx])
        sample = {'features': torch.tensor(self.features[idx], dtype=torch.float32), 
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
    # normalized_dataset = HousePriceTrainDataset(data, selected_features, target_column, normalize_columns)
    train_dataset = HousePriceTrainDataset(data, target_column, normalize_columns)

    # 訪問標準化後的數據集中的樣本
    sample = train_dataset[0]  # 這將顯示標準化後的第一個樣本
    print(sample)