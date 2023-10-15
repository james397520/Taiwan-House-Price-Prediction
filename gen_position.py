import pandas as pd
import requests
import os

# 從文件中讀取 API 密鑰
def get_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# 使用 Google Maps API 反查地址
def reverse_geocode(lat, lng, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
    response = requests.get(url)
    data = response.json()

    if data['status'] == 'OK':
        # 檢查 'plus_code' 和 'compound_code' 是否存在
        plus_code = data.get('plus_code')
        if plus_code and 'compound_code' in plus_code:
            district = plus_code['compound_code'].split(', ')[1][:-3]
            print(district)
            return district
    return None

# API 密鑰
api_key = get_api_key("api_key.txt")  # 請將 "api_key.txt" 替換為您存儲 API 密鑰的文件的實際路徑

# 處理每個 CSV 文件
directory = "data/external_data"  # 請根據實際情況修改目錄名稱
failed_records = pd.DataFrame(columns=["lat", "lng", "file_name"])

for file_name in os.listdir(directory):
    file_path = os.path.join(directory, file_name)
    data = pd.read_csv(file_path)

    # 反查每個座標點的地址
    for index, row in data.iterrows():
        lat, lng = row['lat'], row['lng']
        district = reverse_geocode(lat, lng, api_key)
        if district:
            result = pd.DataFrame({"lat": [lat], "lng": [lng], "file_name": [file_name[:-4]]})
            result.to_csv(f"data/area/{district}.csv", mode='a', header=False, index=False)
        else:
            failed_records.loc[len(failed_records)] = [lat, lng, file_name]

# 將未能成功獲取地址的記錄存儲到一個 CSV 文件中
failed_records.to_csv("data/failed_records.csv", index=False)