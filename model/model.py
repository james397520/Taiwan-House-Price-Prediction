import torch
import torch.nn as nn
import torch.nn.functional as F

class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class TransformerRegressor(nn.Module):
    def __init__(self, feature_size, num_layers, num_heads, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        
        # Creating the positional encoder
        self.positional_encoder = nn.Embedding(feature_size, feature_size)
        
        # Creating the transformer encoder layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, 
            nhead=num_heads, 
            dropout=dropout
        )
        
        # Stacking the transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=num_layers
        )
        
        # Final fully connected layer for regression output
        self.fc = nn.Linear(feature_size, 1)

    def forward(self, x):
        # Adding positional encoding
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        x = x.unsqueeze(1)
        x = x + self.positional_encoder(positions).permute(0, 1, 2)
        
        # Applying the transformer encoder
        x = self.transformer_encoder(x.permute(1, 0, 2))
        
        # Aggregating the output features
        x = x.mean(dim=0)
        
        # Final regression output
        x = self.fc(x)
        return x
