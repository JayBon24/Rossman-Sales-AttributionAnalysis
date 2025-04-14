import pandas as pd
import numpy as np
import os

# 创建结果目录
if not os.path.exists('results'):
    os.makedirs('results')

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

# 生成初步分析报告
print("生成分析报告...")
with open('results/data_explore.md', 'w', encoding='utf-8') as f:
    f.write('# Rossmann销售数据探索\n\n')
    
    f.write('## 1. 数据概览\n\n')
    f.write(f'- 训练数据大小: {train.shape}\n')
    f.write(f'- 店铺数据大小: {store.shape}\n\n')
    
    f.write('## 2. 训练数据样例\n\n')
    f.write('```\n')
    f.write(train.head().to_string())
    f.write('\n```\n\n')
    
    f.write('## 3. 店铺数据样例\n\n')
    f.write('```\n')
    f.write(store.head().to_string())
    f.write('\n```\n\n')
    
    f.write('## 4. 数据字段说明\n\n')
    f.write('### 训练数据字段\n\n')
    for col in train.columns:
        f.write(f'- {col}: {train[col].dtype}\n')
    
    f.write('\n### 店铺数据字段\n\n')
    for col in store.columns:
        f.write(f'- {col}: {store[col].dtype}\n')

print("数据探索完成！结果已保存到results/data_explore.md") 