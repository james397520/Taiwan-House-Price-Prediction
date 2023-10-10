import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
# import zipcodetw
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader



# # Custom dataset
# class HousePriceDataset(Dataset):
#     def __init__(self, csv_file, transform=None):
#         # Load data
#         data = pd.read_csv(csv_file)
#         # district_names = pd.read_csv('data/district_names.csv')
#         # name_code = pd.get_dummies(district_names['行政區名稱'])
#         # pd.DataFrame(name_code)
#         # print(name_code['行政區名稱'].tolist())
        
#         # Drop the '備註' column
#         # data = data.drop(columns=['備註'])
#         # Separate features and target variable

#         self.X = data.iloc[:, :-1].values
#         print(self.X.shape)
#         print(self.X[0])
#         self.ground_size = self.X[:,4].reshape(-1, 1)
#         self.floor = self.X[:,6].reshape(-1, 1)
#         self.all_floor = self.X[:,7].reshape(-1, 1)
#         self.age = self.X[:,10].reshape(-1, 1)
#         self.house_size = self.X[:,11].reshape(-1, 1)
#         self.parking_size = self.X[:,12].reshape(-1, 1)
#         self.parking_cnt = self.X[:,13].reshape(-1, 1)
#         self.lng = self.X[:,14].reshape(-1, 1)
#         self.lat = self.X[:,15].reshape(-1, 1)
#         self.main_size = self.X[:,17].reshape(-1, 1)
#         self.balcony = self.X[:,18].reshape(-1, 1)
#         self.ancillar_size = self.X[:,19].reshape(-1, 1)



#         self.y = data.iloc[:, -1].values.reshape(-1, 1)
#         scaler_x = MinMaxScaler()
#         scaler_y = MinMaxScaler()
#         self.ground_size = scaler_x.fit_transform(self.X[:,4].reshape(-1, 1))
#         # self.X = scaler_x.fit_transform(self.X)
#         self.ground_size = scaler_x.fit_transform(self.X[:,4].reshape(-1, 1))
#         self.y = scaler_y.fit_transform(self.y).flatten()
        
#         self.transform = transform

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         sample = torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
        
#         if self.transform:
#             sample = self.transform(sample)
            
#         return sample

# Z-Score Normalization Function
def z_score_normalize(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Min-Max Normalization Function
def min_max_normalize(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df



# Custom Dataset Class with Normalization Option
class HousePriceDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_column, normalize_columns=None):
        self.dataframe = dataframe.copy()  # Creating a copy to avoid modifying the original dataframe
        
        # Applying the specified normalization methods to the specified columns
        if normalize_columns:
            for column, method in normalize_columns.items():
                if method == 'z-score':
                    self.dataframe = z_score_normalize(self.dataframe, [column])
                elif method == 'min-max':
                    self.dataframe = min_max_normalize(self.dataframe, [column])
                    
        self.features = self.dataframe[feature_columns].values
        self.target = self.dataframe[target_column].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = {'features': torch.tensor(self.features[idx], dtype=torch.float32), 
                  'target': torch.tensor(self.target[idx], dtype=torch.float32)}
        return sample


if __name__ == "__main__":
    # Load dataset
    # train_dataset = HousePriceDataset('data/training_data.csv')
    # print(train_dataset)


    data = pd.read_csv('data/training_data.csv')
    # 指定要標準化的列和標準化方法
    normalize_columns = {
    '土地面積': 'z-score',
    '建物面積': 'min-max'
    }

    # 選擇要用作特徵的列
    selected_features = ['土地面積', '建物面積']
    target_column = '單價'

    # 創建標準化後的數據集
    normalized_dataset = HousePriceDataset(data, selected_features, target_column, normalize_columns)

    # 訪問標準化後的數據集中的樣本
    sample = normalized_dataset[0]  # 這將顯示標準化後的第一個樣本
    print(sample)