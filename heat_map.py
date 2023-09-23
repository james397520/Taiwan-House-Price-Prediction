import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 读取数据
data = pd.read_csv('/Users/jameschan/Project/Taiwan-House-Price-Prediction/data/training_data.csv')  # 请替换 'your_data_file.csv' 为你的数据文件路径

# 提取经纬度数据
longitude = data['橫坐標']
latitude = data['縱坐標']
prices = data['單價']

# 创建热区图
plt.figure(figsize=(12, 8))
plt.scatter(longitude, latitude, c=prices, cmap='coolwarm', s=40, alpha=0.60)
plt.colorbar(label='單價', pad=0.01)  # 根据單價添加颜色条
plt.xlabel('橫坐標')
plt.ylabel('縱坐標')
plt.title('台灣房市熱區圖')
plt.grid(True)

# 可以根据需要保存图像
plt.savefig('heatmap.png')

plt.show()
