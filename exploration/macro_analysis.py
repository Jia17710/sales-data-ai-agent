import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 选几个有代表性的国家，对比能源消费 ============
countries = ["USA", "CHN", "DEU", "IND"]   # 美国、中国、德国、印度
country_names = {"USA": "美国", "CHN": "中国", "DEU": "德国", "IND": "印度"}

# ============ 2. 拉取"人均能源消费量"这个指标，过去20年的数据 ============
data = wb.data.DataFrame("EG.USE.PCAP.KG.OE", countries, mrv=20)

# ============ 1. 数据清洗：转置，让年份变成行，国家变成列（方便画图）============
data_clean = data.T   # .T 表示转置（行列互换）

# 把列名里的"国家代码"换成中文名，方便看
data_clean = data_clean.rename(columns=country_names)

# 把索引里的 "YR2005" 这种格式，去掉"YR"前缀，变成纯数字年份
data_clean.index = data_clean.index.str.replace("YR", "").astype(int)

print(data_clean)
print("---")

# ============ 2. 检查缺失值 ============
print("每列缺失值数量：")
print(data_clean.isna().sum())

# ============ 3. 画趋势对比图 ============
plt.figure(figsize=(12, 6))
for country in country_names.values():
    plt.plot(data_clean.index, data_clean[country], label=country, marker='o')
plt.title("主要国家人均能源消费量趋势对比（千克石油当量）")
plt.xlabel("年份")
plt.ylabel("人均能源消费量")
plt.legend()
plt.savefig("energy_consumption_by_country.png")
print("图表已保存：energy_consumption_by_country.png")