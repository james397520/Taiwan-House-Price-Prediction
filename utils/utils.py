import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


# Function to map coordinates to a feature map with average values
def map_to_feature_map_with_average(coordinates, values, feature_map_size, coordinate_bounds):
    """
    Map coordinates to a feature map with average values.

    :param coordinates: DataFrame with two columns [x, y] for coordinates.
    :param values: Series or list with values to map.
    :param feature_map_size: Tuple (height, width) of the feature map.
    :param coordinate_bounds: Tuple (min_x, max_x, min_y, max_y) of coordinate bounds.
    :return: Torch tensor representing the feature map with average values.
    """
    # Create an empty feature map and a count map
    feature_map = torch.zeros((1, 1, *feature_map_size))
    count_map = torch.zeros((1, 1, *feature_map_size))

    # Normalize coordinates to fit within the feature map
    min_x, max_x, min_y, max_y = coordinate_bounds
    coordinates['norm_x'] = (coordinates['橫坐標'] - min_x) / (max_x - min_x) * (feature_map_size[1] - 1)
    coordinates['norm_y'] = (coordinates['縱坐標'] - min_y) / (max_y - min_y) * (feature_map_size[0] - 1)

    for index, row in coordinates.iterrows():
        x = int(row['norm_x'])
        y = int(row['norm_y'])
        value = values.iloc[index]

        feature_map[0, 0, y, x] += value
        count_map[0, 0, y, x] += 1

    # Calculate the average
    feature_map /= count_map.clamp(min=1)

    return feature_map


# Function to map one-hot encoded district IDs to a feature map
def map_district_to_feature_map(coordinates, one_hot_values, feature_map_size, coordinate_bounds):
    """
    Map one-hot encoded district IDs to a feature map.

    :param coordinates: DataFrame with two columns [x, y] for coordinates.
    :param one_hot_values: Numpy array with one-hot encoded values.
    :param feature_map_size: Tuple (height, width) of the feature map.
    :param coordinate_bounds: Tuple (min_x, max_x, min_y, max_y) of coordinate bounds.
    :return: Torch tensor representing the feature map for district IDs.
    """
    num_classes = one_hot_values.shape[1]
    feature_map = torch.zeros((1, num_classes, *feature_map_size))

    # Normalize coordinates
    min_x, max_x, min_y, max_y = coordinate_bounds
    coordinates['norm_x'] = (coordinates['橫坐標'] - min_x) / (max_x - min_x) * (feature_map_size[1] - 1)
    coordinates['norm_y'] = (coordinates['縱坐標'] - min_y) / (max_y - min_y) * (feature_map_size[0] - 1)

    for index, row in coordinates.iterrows():
        x = int(row['norm_x'])
        y = int(row['norm_y'])
        feature_map[0, :, y, x] = torch.tensor(one_hot_values[index])

    return feature_map






if __name__ == "__main__":

    print("QQ")

    # # Map the coordinates to a feature map with average values
    # feature_map_avg = map_to_feature_map_with_average(final_training_data[['橫坐標', '縱坐標']], final_training_data['單價'], feature_map_size, (min_x, max_x, min_y, max_y))

    # feature_map_avg.shape, feature_map_avg.isnan().any()

    # #One hot encode the district IDs
    # encoder = OneHotEncoder(sparse=False)
    # district_ids = final_training_data[['行政區編號']].values
    # one_hot_district_ids = encoder.fit_transform(district_ids)
    # # Map the district IDs to a feature map
    # feature_map_districts = map_district_to_feature_map(final_training_data[['橫坐標', '縱坐標']], one_hot_district_ids, feature_map_size, (min_x, max_x, min_y, max_y))

    # feature_map_districts.shape
    