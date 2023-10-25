import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch

# Load GeoJSON and data
taiwan_map = gpd.read_file('path_to_geojson_file.geo.json')
data = pd.read_csv('path_to_data_file.csv')

# Calculate the average price for each county
average_prices = data.groupby('縣市')['單價'].mean()

# Merge the average prices with the taiwan_map GeoDataFrame
taiwan_map_merged = taiwan_map.merge(average_prices, left_on='COUNTYNAME', right_index=True, how='left')

# Convert the plot to a numpy array and resize it to 640x640
image = np.array(taiwan_map_merged['單價'].fillna(0).values).reshape((22, 1))
resized_image = resize(image, (640, 640), mode='reflect', anti_aliasing=True)

# Normalize the image values to range 0-1
normalized_image = (resized_image - resized_image.min()) / (resized_image.max() - resized_image.min())

# Convert the numpy array to a PyTorch tensor with shape 1x1x640x640 (if needed)
tensor_map = torch.tensor(normalized_image).unsqueeze(0).unsqueeze(0).float()

# Display the feature map
plt.imshow(normalized_image, cmap='plasma')
plt.colorbar()
plt.title('Feature Map with Average Prices by County')
plt.show()
