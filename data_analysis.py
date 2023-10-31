import pandas as pd
import matplotlib.pyplot as plt
import os

# 讀取資料
data = pd.read_csv('data/final_updated_training_data.csv')

# 中英特徵名稱對照表
translation_map = {
    '行政區編號': 'District_ID',
    '土地面積': 'Land_Area',
    '使用分區': 'Usage_Zone',
    '移轉層次': 'Transfer_Level',
    '總樓層數': 'Total_Floors',
    '主要用途': 'Main_Usage',
    '主要建材': 'Main_Material',
    '建物型態': 'Building_Type',
    '屋齡': 'Building_Age',
    '建物面積': 'Building_Area',
    '車位面積': 'Parking_Area',
    '車位個數': 'Parking_Count',
    '橫坐標': 'Latitude',
    '縱坐標': 'Longitude',
    '主建物面積': 'Main_Building_Area',
    '陽台面積': 'Balcony_Area',
    '附屬建物面積': 'Attached_Building_Area'
}

# 選擇的特徵
selected_features = [
    '行政區編號', '土地面積', '使用分區', '移轉層次', '總樓層數', '主要用途', '主要建材',
    '建物型態', '屋齡', '建物面積', '車位面積', '車位個數', '橫坐標', '縱坐標', '主建物面積', '陽台面積', '附屬建物面積'
]

# 建立一個資料夾來存儲散點圖
if not os.path.exists('scatter_plots'):
    os.makedirs('scatter_plots')

# 製作並存儲散點圖
for feature in selected_features:
    plt.figure(figsize=(12, 6))
    english_feature_name = translation_map.get(feature, feature)
    plt.scatter(data[feature], data["單價"], alpha=0.5)
    plt.title(f'{english_feature_name} vs Unit Price')
    plt.xlabel(english_feature_name)
    plt.ylabel('Unit Price')
    plt.tight_layout()
    plt.savefig(f'./scatter_plots/{english_feature_name}_vs_UnitPrice.png')
    plt.close()
