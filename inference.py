
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import HousePriceModel
from dataloader import  HousePriceTestDataset, min_max_denormalize, z_score_denormalize
import platform



def inference():
    # Load Data
    data_path = 'data/public_dataset.csv'  # Update with the path of your data file
    data = pd.read_csv(data_path)
    normalize_columns = {'橫坐標': 'min-max', '縱坐標': 'min-max'}
    selected_features = ['橫坐標', '縱坐標']
    if platform.system() == "Linux":
        gpu = True
    elif platform.system() == "Darwin":
        gpu = False

    dataset = HousePriceTestDataset(data, selected_features, normalize_columns)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Load Model
    model_path = 'model.pth'  # Update with the path of your trained model file
    input_dim = len(selected_features)
    if gpu:
        model = HousePriceModel(input_dim).cuda()
    else:
        model = HousePriceModel(input_dim)
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
    predicted_prices_df = min_max_denormalize(predicted_prices_df, ["predicted_price"],scaler_path="label_min_max_normalize_data.pkl")
    # Save to CSV
    output_csv_path = 'predicted_prices.csv'  # Update with the desired output path
    predicted_prices_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

if __name__ == "__main__":
    inference()