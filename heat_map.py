import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 讀取資料
data = pd.read_csv('data/training_data.csv')  # 請替換 'your_data_file.csv' 為你的資料文件路徑

# 提取經緯度資料
longitude = data['橫坐標']
latitude = data['縱坐標']
prices = data['單價']

# 創建熱區圖    
plt.figure(figsize=(12, 8))
plt.scatter(longitude, latitude, c=prices, cmap='coolwarm', s=40, alpha=0.60)
plt.colorbar(label='單價', pad=0.01)  # 根據單價添加顏色條
plt.xlabel('橫坐標')
plt.ylabel('縱坐標')
plt.title('台灣房市熱區圖')
plt.grid(True)

# 可以根據需要保存圖像
plt.savefig('heatmap.png')

plt.show()
