# # import torch
# # from sklearn.preprocessing import StandardScaler
# # import pandas as pd
# # import torch
# # from torch.utils.data import Dataset
# # from sklearn.preprocessing import MinMaxScaler

# # # Custom dataset
# # class HousePriceDataset(Dataset):
# #     def __init__(self, csv_file):
# #         data = pd.read_csv(csv_file)
# #         self.X = data.iloc[:, :-1].values
# #         self.y = data.iloc[:, -1].values

# #         # Standardize features
# #         scaler = StandardScaler()
# #         self.X = scaler.fit_transform(self.X)

# #     def __len__(self):
# #         return len(self.y)

# #     def __getitem__(self, idx):
# #         return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)



import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import zipcodetw



# Custom dataset
class HousePriceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Load data
        data = pd.read_csv(csv_file)
        # district_names = pd.read_csv('data/district_names.csv')
        # name_code = pd.get_dummies(district_names['行政區名稱'])
        # pd.DataFrame(name_code)
        # print(name_code['行政區名稱'].tolist())
        
        # Drop the '備註' column
        data = data.drop(columns=['備註'])
        # Separate features and target variable

        self.X = data.iloc[:, :-1].values
        print(self.X.shape)
        print(self.X[0])
        self.size = self.X[:,4]
        print("OG SIZE: ",self.size[0])
        print(self.size)
        # data['完整地址'] = data['縣市'] + data['鄉鎮市區'] + data['路名']
        # print(data.head)
        # print("QQQ: ",zipcodetw.find('臺北市大安區敦化南路二段'))
        # data_dum = pd.get_dummies(data['主要用途'])
        # pd.DataFrame(data_dum)
        # print(data_dum.head)

        self.y = data.iloc[:, -1].values.reshape(-1, 1)
        print(type(self.X[:,4]))
        # Apply Min-Max normalization
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        # self.X = scaler_x.fit_transform(self.X)
        self.size = scaler_x.fit_transform(self.X[:,4].reshape(-1, 1))
        print("fit_transform SIZE: ",self.size)
        self.y = scaler_y.fit_transform(self.y).flatten()
        
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import OneHotEncoder
# import zipcodetw

# class HousePriceDataset(Dataset):
#     def __init__(self, csv_file):
#         # Load the dataset
#         self.dataset = pd.read_csv(csv_file)

#         # Specify the columns to be one-hot encoded
#         # self.one_hot_columns = one_hot_columns

#         # Initialize OneHotEncoder
#         # self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
#         # self.encoder.fit(self.dataset[one_hot_columns])

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         # Get the data point
#         data_point = self.dataset.iloc[idx]
#         print(data_point)
#         # Extract the values to be one-hot encoded
#         # values_to_encode = data_point[self.one_hot_columns].values.reshape(1, -1)
        
#         # Perform one-hot encoding
#         # one_hot_encoded = self.encoder.transform(values_to_encode)

#         # Convert to tensor
#         # one_hot_tensor = torch.tensor(one_hot_encoded, dtype=torch.float32)

#         return data_point




if __name__ == "__main__":
    # Load dataset
    train_dataset = HousePriceDataset('data/training_data.csv')
    print(train_dataset)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # for batch_idx, (data, targets) in enumerate(train_loader):
    #     print("QQdata",data.shape)
    #     print("TTARGET",targets.shape)

    # Usage example
    # one_hot_columns = ['縣市']  # Specify the columns to be one-hot encoded
    # dataset = HousePriceDataset('data/training_data.csv', one_hot_columns)

    # Print the one-hot encoded features of the first data point
    # print(dataset[0].shape)