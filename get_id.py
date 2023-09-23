import pandas as pd

# 读取数据
data = pd.read_csv('data/training_data.csv')  # 请替换 'your_data_file.csv' 为你的数据文件路径

# 提取縣市、鄉鎮市區和路名列
counties = data['縣市'].unique()
towns = data['鄉鎮市區'].unique()
roads = data['路名'].unique()

# 创建字典来存储縣市、鄉鎮市區和路名的编号
county_dict = {county: i for i, county in enumerate(counties, start=1)}
town_dict = {town: i for i, town in enumerate(towns, start=1)}
road_dict = {road: i for i, road in enumerate(roads, start=1)}

# 将编号添加到数据中
data['縣市編號'] = data['縣市'].map(county_dict)
data['鄉鎮市區編號'] = data['鄉鎮市區'].map(town_dict)
data['路名編號'] = data['路名'].map(road_dict)

# 打印前几行数据，包含编号
print(data[['縣市', '縣市編號', '鄉鎮市區', '鄉鎮市區編號', '路名', '路名編號']].head())

# 可以根据需要保存修改后的数据
data.to_csv('data_with_id.csv', index=False)
