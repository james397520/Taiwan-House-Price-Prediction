
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model.model import HousePriceModel, TransformerRegressor
from dataloader import  HousePriceTestDataset, min_max_denormalize, z_score_denormalize
import platform



def inference():
    # Load Data
    data_path = 'data/public_dataset.csv'  # Update with the path of your data file
    data = pd.read_csv(data_path)
    # normalize_columns = {'橫坐標': 'min-max', '縱坐標': 'min-max'}
    normalize_columns = {
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

    if platform.system() == "Linux":
        gpu = True
    elif platform.system() == "Darwin":
        gpu = False
    
    dataset = HousePriceTestDataset(data, normalize_columns)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Load Model
    model_path = 'model.pth'  # Update with the path of your trained model file
    input_dim = len(normalize_columns.keys())
    model = HousePriceModel(input_dim)
    model = TransformerRegressor(input_dim, 4, 6)
    if gpu:
        model = model.cuda()
    else:
        model = model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Inference
    with torch.no_grad():
        for batch in data_loader:
            if gpu:
                features = batch['features'].cuda()
            else:
                features = batch['features']
            
            predictions = model(features).cpu().numpy().flatten()

    # Create a DataFrame to hold the IDs and predicted prices
    ids = [f"PU-{i}" for i in range(1, len(predictions) + 1)]  # Adjust ID format as needed
    predicted_prices_df = pd.DataFrame({"ID": ids, "predicted_price": predictions})
    predicted_prices_df = min_max_denormalize(predicted_prices_df, ["predicted_price"],scaler_path="pkl/單價_min_max_normalize_data.pkl")
    # Save to CSV
    output_csv_path = 'predicted_prices.csv'  # Update with the desired output path
    predicted_prices_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

if __name__ == "__main__":
    inference()