
import json
import numpy as np



# Function to save scaler parameters to a JSON file
def save_scaler_params(scaler, path):
    params = {
        "mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
        "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
        "min": scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
        "max": scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None
    }
    with open(path, 'w') as file:
        json.dump(params, file)

# Function to load scaler parameters from a JSON file
def load_scaler_params(path):
    with open(path, 'r') as file:
        params = json.load(file)
    return params



class HousePriceTestDataset(Dataset):
    def __init__(self, dataframe, feature_columns, normalize_columns=None):
        self.dataframe = dataframe.copy()  # Creating a copy to avoid modifying the original dataframe
        
        # Applying the specified normalization methods to the specified columns
        if normalize_columns:
            for column, method in normalize_columns.items():
                if method == 'z-score':
                    self.dataframe = z_score_normalize(self.dataframe, [column])
                elif method == 'min-max':
                    self.dataframe = min_max_normalize(self.dataframe, [column])

