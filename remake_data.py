import pandas as pd
from typing import Dict, List, Union

def load_and_combine_csv(filepaths: List[str]) -> pd.DataFrame:
    """
    Load and combine multiple CSV files into a single DataFrame.

    Parameters:
    - filepaths: A list of filepaths of the CSV files to be loaded and combined.

    Returns:
    - A combined DataFrame containing data from all provided CSV files.
    """
    dataframes = [pd.read_csv(filepath)[['縣市', '鄉鎮市區']] for filepath in filepaths]
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

def create_county_township_dict(data: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Create a dictionary mapping counties to their townships.

    Parameters:
    - data: A DataFrame containing columns '縣市' and '鄉鎮市區'.

    Returns:
    - A dictionary with counties as keys and lists of townships as values.
    """
    counties_townships_dict = {}
    
    for _, row in data.iterrows():
        county = row['縣市']
        township = row['鄉鎮市區']
        
        if county not in counties_townships_dict:
            counties_townships_dict[county] = set()
        counties_townships_dict[county].add(township)

    # Convert sets to lists for easier readability and usage
    counties_townships_dict = {county: list(townships) for county, townships in counties_townships_dict.items()}
    
    return counties_townships_dict

# Filepaths of the CSV files to be loaded
filepaths = ['data/training_data.csv', 'data/public_dataset.csv']

# Load and combine the CSV files
combined_data = load_and_combine_csv(filepaths)

# Create a dictionary mapping counties to their townships
counties_townships_dict = create_county_township_dict(combined_data)

# Display the first few items in the dictionary to check the data
area_dict = dict(list(counties_townships_dict.items())), len(counties_townships_dict)
print(counties_townships_dict.keys())