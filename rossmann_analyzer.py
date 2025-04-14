import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建结果目录
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/figures'):
    os.makedirs('results/figures')

# 读取数据
print("开始读取数据...")
train = pd.read_csv('datasets/train.csv')
store = pd.read_csv('datasets/store.csv')

print("数据基本信息:")
print(f"训练数据大小: {train.shape}")
print(f"店铺数据大小: {store.shape}")

# 查看训练数据的前几行
print("\n训练数据前5行:")
print(train.head())

# 查看店铺数据的前几行
print("\n店铺数据前5行:")
print(store.head())

# 合并数据
print("合并数据...")
data = pd.merge(train, store, on='Store', how='left')

# 转换日期格式
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek + 1  # 确保与原始数据一致
data['WeekOfYear'] = data['Date'].dt.isocalendar().week

# 计算月销售总额
print("计算月销售总额...")
monthly_sales = data.groupby(['Year', 'Month']).agg({'Sales': 'sum'}).reset_index()
monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str).str.zfill(2)
monthly_sales = monthly_sales.sort_values(['Year', 'Month'])

# 计算环比增长率
monthly_sales['Sales_MoM'] = monthly_sales['Sales'].pct_change()

# 只取最近三个月的数据
recent_months = monthly_sales.tail(3).copy()
recent_months['YearMonth'] = pd.to_datetime(recent_months['YearMonth'] + '-01').dt.strftime('%Y-%m')

print("\n最近三个月销售情况:")
print(recent_months[['YearMonth', 'Sales', 'Sales_MoM']])

# 保存最近三个月销售趋势图
plt.figure(figsize=(10, 6))
plt.plot(recent_months['YearMonth'], recent_months['Sales'], marker='o')
plt.title('最近三个月销售额趋势')
plt.xlabel('年月')
plt.ylabel('销售额')
plt.grid(True)
plt.savefig('results/figures/monthly_sales.png')
plt.close()

# 获取最近三个月的年月信息
last_three_months = monthly_sales.tail(3)['YearMonth'].tolist()
last_three_months_data = data[data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2).isin(last_three_months)]

print(f"\n最近三个月的数据量: {last_three_months_data.shape}")

# 1. 按店铺类型分析销售额
print("按店铺类型分析销售额...")
store_type_sales = last_three_months_data.groupby(['Year', 'Month', 'StoreType']).agg({'Sales': 'sum'}).reset_index()
store_type_sales['YearMonth'] = store_type_sales['Year'].astype(str) + '-' + store_type_sales['Month'].astype(str).str.zfill(2)

# 透视表便于计算环比
store_type_pivot = store_type_sales.pivot_table(index='StoreType', columns='YearMonth', values='Sales')
print("\n按店铺类型的销售额:")
print(store_type_pivot)

# 生成初步分析报告
print("生成分析报告...")
with open('results/basic_analysis.md', 'w', encoding='utf-8') as f:
    f.write('# Rossmann销售数据环比增长归因分析-初步报告\n\n')
    
    f.write('## 1. 数据概览\n\n')
    f.write(f'- 训练数据大小: {train.shape}\n')
    f.write(f'- 店铺数据大小: {store.shape}\n')
    f.write(f'- 合并后数据大小: {data.shape}\n\n')
    
    f.write('## 2. 最近三个月销售概况\n\n')
    f.write('最近三个月销售额及环比变化：\n\n')
    f.write('| 年月 | 销售额 | 环比变化 |\n')
    f.write('| --- | --- | --- |\n')
    for _, row in recent_months.iterrows():
        f.write(f"| {row['YearMonth']} | {row['Sales']:,.0f} | {row['Sales_MoM']:.2%} |\n")
    f.write('\n')
    
    f.write('## 3. 店铺类型销售分析\n\n')
    f.write('不同店铺类型的销售额：\n\n')
    f.write('| 店铺类型 | ' + ' | '.join(store_type_pivot.columns.astype(str)) + ' |\n')
    f.write('| --- | ' + ' | '.join(['---' for _ in range(len(store_type_pivot.columns))]) + ' |\n')
    for idx, row in store_type_pivot.iterrows():
        row_values = []
        for col in store_type_pivot.columns:
            row_values.append(f"{row[col]:,.0f}")
        f.write(f"| {idx} | " + " | ".join(row_values) + " |\n")

print("初步分析完成！结果已保存到results目录下。") 