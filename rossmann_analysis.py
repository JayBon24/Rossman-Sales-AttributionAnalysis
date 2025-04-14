import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.ticker as mtick

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置标准输出编码为 UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取数据
print("开始读取数据...")
train = pd.read_csv('datasets/train.csv')
store = pd.read_csv('datasets/store.csv')

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

# 创建结果目录
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/figures'):
    os.makedirs('results/figures')

print("数据基本信息:")
print(f"训练数据大小: {train.shape}")
print(f"店铺数据大小: {store.shape}")
print(f"合并后数据大小: {data.shape}")

# 输出数据的基本信息
print("\n数据前5行:")
print(data.head())

print("\n数据描述性统计:")
print(data.describe())

print("\n缺失值情况:")
print(data.isnull().sum())

# 填充缺失值
data['CompetitionDistance'].fillna(data['CompetitionDistance'].median(), inplace=True)
for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']:
    data[col].fillna(0, inplace=True)
data['PromoInterval'].fillna('', inplace=True)

# 计算月销售总额
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

# 获取最近三个月的年月信息
last_three_months = monthly_sales.tail(3)['YearMonth'].tolist()
last_three_months_data = data[data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2).isin(last_three_months)]

print(f"\n最近三个月的数据量: {last_three_months_data.shape}")

# 1. 按店铺类型分析销售额
store_type_sales = last_three_months_data.groupby(['Year', 'Month', 'StoreType']).agg({'Sales': 'sum'}).reset_index()
store_type_sales['YearMonth'] = store_type_sales['Year'].astype(str) + '-' + store_type_sales['Month'].astype(str).str.zfill(2)

# 透视表便于计算环比
store_type_pivot = store_type_sales.pivot_table(index='StoreType', columns='YearMonth', values='Sales')
store_type_pivot['环比变化1'] = store_type_pivot[store_type_pivot.columns[1]] / store_type_pivot[store_type_pivot.columns[0]] - 1
store_type_pivot['环比变化2'] = store_type_pivot[store_type_pivot.columns[2]] / store_type_pivot[store_type_pivot.columns[1]] - 1

print("\n按店铺类型的销售额环比:")
print(store_type_pivot)

# 绘制店铺类型销售额环比变化图
plt.figure(figsize=(10, 6))
for st in store_type_sales['StoreType'].unique():
    subset = store_type_sales[store_type_sales['StoreType'] == st]
    plt.plot(subset['YearMonth'], subset['Sales'], marker='o', label=f'类型{st}')
plt.title('不同店铺类型最近三个月销售额变化')
plt.xlabel('年月')
plt.ylabel('销售额')
plt.legend()
plt.grid(True)
plt.savefig('results/figures/store_type_sales.png')
plt.close()

# 2. 按促销活动分析销售额
promo_sales = last_three_months_data.groupby(['Year', 'Month', 'Promo']).agg({'Sales': 'sum'}).reset_index()
promo_sales['YearMonth'] = promo_sales['Year'].astype(str) + '-' + promo_sales['Month'].astype(str).str.zfill(2)

promo_pivot = promo_sales.pivot_table(index='Promo', columns='YearMonth', values='Sales')
promo_pivot['环比变化1'] = promo_pivot[promo_pivot.columns[1]] / promo_pivot[promo_pivot.columns[0]] - 1
promo_pivot['环比变化2'] = promo_pivot[promo_pivot.columns[2]] / promo_pivot[promo_pivot.columns[1]] - 1

print("\n按促销活动的销售额环比:")
print(promo_pivot)

# 绘制促销活动销售额环比变化图
plt.figure(figsize=(10, 6))
for p in promo_sales['Promo'].unique():
    subset = promo_sales[promo_sales['Promo'] == p]
    plt.plot(subset['YearMonth'], subset['Sales'], marker='o', label=f'促销={p}')
plt.title('促销活动最近三个月销售额变化')
plt.xlabel('年月')
plt.ylabel('销售额')
plt.legend()
plt.grid(True)
plt.savefig('results/figures/promo_sales.png')
plt.close()

# 3. 按商品种类分析销售额
assortment_sales = last_three_months_data.groupby(['Year', 'Month', 'Assortment']).agg({'Sales': 'sum'}).reset_index()
assortment_sales['YearMonth'] = assortment_sales['Year'].astype(str) + '-' + assortment_sales['Month'].astype(str).str.zfill(2)

assortment_pivot = assortment_sales.pivot_table(index='Assortment', columns='YearMonth', values='Sales')
assortment_pivot['环比变化1'] = assortment_pivot[assortment_pivot.columns[1]] / assortment_pivot[assortment_pivot.columns[0]] - 1
assortment_pivot['环比变化2'] = assortment_pivot[assortment_pivot.columns[2]] / assortment_pivot[assortment_pivot.columns[1]] - 1

print("\n按商品种类的销售额环比:")
print(assortment_pivot)

# 绘制商品种类销售额环比变化图
plt.figure(figsize=(10, 6))
for a in assortment_sales['Assortment'].unique():
    subset = assortment_sales[assortment_sales['Assortment'] == a]
    plt.plot(subset['YearMonth'], subset['Sales'], marker='o', label=f'商品种类={a}')
plt.title('不同商品种类最近三个月销售额变化')
plt.xlabel('年月')
plt.ylabel('销售额')
plt.legend()
plt.grid(True)
plt.savefig('results/figures/assortment_sales.png')
plt.close()

# 4. 按星期几分析销售额
dow_sales = last_three_months_data.groupby(['Year', 'Month', 'DayOfWeek']).agg({'Sales': 'sum'}).reset_index()
dow_sales['YearMonth'] = dow_sales['Year'].astype(str) + '-' + dow_sales['Month'].astype(str).str.zfill(2)

# 绘制星期几销售额图
plt.figure(figsize=(12, 6))
for ym in dow_sales['YearMonth'].unique():
    subset = dow_sales[dow_sales['YearMonth'] == ym]
    plt.plot(subset['DayOfWeek'], subset['Sales'], marker='o', label=ym)
plt.title('不同星期几的销售额变化')
plt.xlabel('星期几')
plt.ylabel('销售额')
plt.xticks(range(1, 8))
plt.legend()
plt.grid(True)
plt.savefig('results/figures/dow_sales.png')
plt.close()

# 5. 假日影响分析
holiday_sales = last_three_months_data.groupby(['Year', 'Month', 'SchoolHoliday']).agg({'Sales': 'sum'}).reset_index()
holiday_sales['YearMonth'] = holiday_sales['Year'].astype(str) + '-' + holiday_sales['Month'].astype(str).str.zfill(2)

holiday_pivot = holiday_sales.pivot_table(index='SchoolHoliday', columns='YearMonth', values='Sales')
holiday_pivot['环比变化1'] = holiday_pivot[holiday_pivot.columns[1]] / holiday_pivot[holiday_pivot.columns[0]] - 1
holiday_pivot['环比变化2'] = holiday_pivot[holiday_pivot.columns[2]] / holiday_pivot[holiday_pivot.columns[1]] - 1

print("\n按学校假期的销售额环比:")
print(holiday_pivot)

# 绘制假日销售额环比变化图
plt.figure(figsize=(10, 6))
for h in holiday_sales['SchoolHoliday'].unique():
    subset = holiday_sales[holiday_sales['SchoolHoliday'] == h]
    plt.plot(subset['YearMonth'], subset['Sales'], marker='o', label=f'学校假期={h}')
plt.title('学校假期最近三个月销售额变化')
plt.xlabel('年月')
plt.ylabel('销售额')
plt.legend()
plt.grid(True)
plt.savefig('results/figures/holiday_sales.png')
plt.close()

# 6. 竞争对手距离对销售的影响
# 将距离分为几个区间
data['CompetitionDistanceBin'] = pd.cut(data['CompetitionDistance'], 
                                        bins=[0, 1000, 5000, 10000, 20000, np.inf], 
                                        labels=['<1km', '1-5km', '5-10km', '10-20km', '>20km'])

competition_sales = last_three_months_data.groupby(['Year', 'Month', 'CompetitionDistanceBin']).agg({'Sales': 'sum'}).reset_index()
competition_sales['YearMonth'] = competition_sales['Year'].astype(str) + '-' + competition_sales['Month'].astype(str).str.zfill(2)

competition_pivot = competition_sales.pivot_table(index='CompetitionDistanceBin', columns='YearMonth', values='Sales')
competition_pivot['环比变化1'] = competition_pivot[competition_pivot.columns[1]] / competition_pivot[competition_pivot.columns[0]] - 1
competition_pivot['环比变化2'] = competition_pivot[competition_pivot.columns[2]] / competition_pivot[competition_pivot.columns[1]] - 1

print("\n按竞争对手距离的销售额环比:")
print(competition_pivot)

# 7. 随机森林模型进行特征重要性分析
print("\n执行特征重要性分析...")

# 准备建模数据
model_data = last_three_months_data.copy()

# 编码分类特征
le_store_type = LabelEncoder()
le_assortment = LabelEncoder()
model_data['StoreType_encoded'] = le_store_type.fit_transform(model_data['StoreType'])
model_data['Assortment_encoded'] = le_assortment.fit_transform(model_data['Assortment'])

# 选择特征
features = [
    'DayOfWeek', 'Promo', 'SchoolHoliday', 'StoreType_encoded', 
    'Assortment_encoded', 'CompetitionDistance', 'Promo2'
]

X = model_data[features]
y = model_data['Sales']

# 训练随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# 计算特征重要性
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\n特征重要性:")
print(feature_importance)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('影响销售额的因素重要性')
plt.tight_layout()
plt.savefig('results/figures/feature_importance.png')
plt.close()

# 8. 各因素对环比变化的贡献度分析
print("\n分析各因素对环比变化的贡献度...")

# 计算每个月的平均销售额
monthly_avg = last_three_months_data.groupby(['Year', 'Month']).agg({'Sales': 'mean'}).reset_index()
monthly_avg['YearMonth'] = monthly_avg['Year'].astype(str) + '-' + monthly_avg['Month'].astype(str).str.zfill(2)

# 计算环比变化
monthly_avg['Sales_MoM'] = monthly_avg['Sales'].pct_change()

# 获取最后两个月的环比变化
last_mom = monthly_avg.iloc[-1]['Sales_MoM']
print(f"最后一个月的环比变化: {last_mom:.2%}")

# 计算各因素对环比变化的贡献
factors = ['StoreType', 'Assortment', 'Promo', 'SchoolHoliday', 'CompetitionDistanceBin']
contribution_results = []

for factor in factors:
    factor_monthly = last_three_months_data.groupby(['Year', 'Month', factor]).agg({'Sales': 'sum'}).reset_index()
    factor_monthly['YearMonth'] = factor_monthly['Year'].astype(str) + '-' + factor_monthly['Month'].astype(str).str.zfill(2)
    
    # 计算每个因素类别的月度销售额占比
    total_monthly = factor_monthly.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
    factor_monthly = factor_monthly.merge(total_monthly, on=['Year', 'Month'], suffixes=('', '_total'))
    factor_monthly['Sales_Ratio'] = factor_monthly['Sales'] / factor_monthly['Sales_total']
    
    # 计算每个因素类别最后两个月的占比变化
    factor_pivot = factor_monthly.pivot_table(index=factor, columns='YearMonth', values='Sales_Ratio')
    
    # 计算贡献度 - 简化方法：最后一个月的占比变化 * 最后一个月的总体环比变化
    if len(factor_pivot.columns) >= 2:
        factor_pivot['占比变化'] = factor_pivot[factor_pivot.columns[-1]] - factor_pivot[factor_pivot.columns[-2]]
        factor_pivot['贡献度'] = factor_pivot['占比变化'] * last_mom
        
        # 保存结果
        for idx, row in factor_pivot.iterrows():
            if 'Ratio' in factor_pivot.columns and '占比变化' in factor_pivot.columns and '贡献度' in factor_pivot.columns:
                contribution_results.append({
                    '因素': factor,
                    '类别': idx,
                    '上月占比': row[factor_pivot.columns[-2]] if not pd.isna(row[factor_pivot.columns[-2]]) else 0,
                    '当月占比': row[factor_pivot.columns[-1]] if not pd.isna(row[factor_pivot.columns[-1]]) else 0,
                    '占比变化': row['占比变化'] if not pd.isna(row['占比变化']) else 0,
                    '贡献度': row['贡献度'] if not pd.isna(row['贡献度']) else 0
                })

# 转换为DataFrame
contribution_df = pd.DataFrame(contribution_results)
if not contribution_df.empty:
    # 按贡献度绝对值排序
    contribution_df['贡献度绝对值'] = contribution_df['贡献度'].abs()
    contribution_df = contribution_df.sort_values('贡献度绝对值', ascending=False)
    
    print("\n各因素对环比变化的贡献度:")
    print(contribution_df[['因素', '类别', '上月占比', '当月占比', '占比变化', '贡献度']])
    
    # 绘制贡献度瀑布图 (简化版)
    plt.figure(figsize=(12, 8))
    # 只取贡献度绝对值前10的因素
    top_contribution = contribution_df.head(10)
    colors = ['green' if x > 0 else 'red' for x in top_contribution['贡献度']]
    
    sns.barplot(x='贡献度', y=top_contribution['因素'] + '-' + top_contribution['类别'].astype(str), 
                data=top_contribution, palette=colors)
    plt.title('各因素对销售额环比变化的贡献度(Top 10)')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('results/figures/contribution_waterfall.png')
    plt.close()

# 生成分析报告
with open('results/analysis_report.md', 'w', encoding='utf-8') as f:
    f.write('# Rossmann销售数据环比增长归因分析报告\n\n')
    
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
    
    f.write('## 3. 各维度对销售额的影响\n\n')
    
    # 店铺类型影响
    f.write('### 3.1 店铺类型影响\n\n')
    f.write('不同店铺类型的销售额环比变化：\n\n')
    f.write('| 店铺类型 | ' + ' | '.join(store_type_pivot.columns.astype(str)) + ' |\n')
    f.write('| --- | ' + ' | '.join(['---' for _ in range(len(store_type_pivot.columns))]) + ' |\n')
    for idx, row in store_type_pivot.iterrows():
        row_values = []
        for col in store_type_pivot.columns:
            if col.startswith('环比变化'):
                row_values.append(f"{row[col]:.2%}")
            else:
                row_values.append(f"{row[col]:,.0f}")
        f.write(f"| {idx} | " + " | ".join(row_values) + " |\n")
    f.write('\n![店铺类型销售额变化](figures/store_type_sales.png)\n\n')
    
    # 促销活动影响
    f.write('### 3.2 促销活动影响\n\n')
    f.write('促销活动的销售额环比变化：\n\n')
    f.write('| 促销活动 | ' + ' | '.join(promo_pivot.columns.astype(str)) + ' |\n')
    f.write('| --- | ' + ' | '.join(['---' for _ in range(len(promo_pivot.columns))]) + ' |\n')
    for idx, row in promo_pivot.iterrows():
        row_values = []
        for col in promo_pivot.columns:
            if col.startswith('环比变化'):
                row_values.append(f"{row[col]:.2%}")
            else:
                row_values.append(f"{row[col]:,.0f}")
        f.write(f"| {idx} | " + " | ".join(row_values) + " |\n")
    f.write('\n![促销活动销售额变化](figures/promo_sales.png)\n\n')
    
    # 商品种类影响
    f.write('### 3.3 商品种类影响\n\n')
    f.write('不同商品种类的销售额环比变化：\n\n')
    f.write('| 商品种类 | ' + ' | '.join(assortment_pivot.columns.astype(str)) + ' |\n')
    f.write('| --- | ' + ' | '.join(['---' for _ in range(len(assortment_pivot.columns))]) + ' |\n')
    for idx, row in assortment_pivot.iterrows():
        row_values = []
        for col in assortment_pivot.columns:
            if col.startswith('环比变化'):
                row_values.append(f"{row[col]:.2%}")
            else:
                row_values.append(f"{row[col]:,.0f}")
        f.write(f"| {idx} | " + " | ".join(row_values) + " |\n")
    f.write('\n![商品种类销售额变化](figures/assortment_sales.png)\n\n')
    
    # 学校假期影响
    f.write('### 3.4 学校假期影响\n\n')
    f.write('学校假期的销售额环比变化：\n\n')
    f.write('| 学校假期 | ' + ' | '.join(holiday_pivot.columns.astype(str)) + ' |\n')
    f.write('| --- | ' + ' | '.join(['---' for _ in range(len(holiday_pivot.columns))]) + ' |\n')
    for idx, row in holiday_pivot.iterrows():
        row_values = []
        for col in holiday_pivot.columns:
            if col.startswith('环比变化'):
                row_values.append(f"{row[col]:.2%}")
            else:
                row_values.append(f"{row[col]:,.0f}")
        f.write(f"| {idx} | " + " | ".join(row_values) + " |\n")
    f.write('\n![学校假期销售额变化](figures/holiday_sales.png)\n\n')
    
    # 竞争对手距离影响
    f.write('### 3.5 竞争对手距离影响\n\n')
    f.write('不同竞争对手距离的销售额环比变化：\n\n')
    f.write('| 竞争对手距离 | ' + ' | '.join(competition_pivot.columns.astype(str)) + ' |\n')
    f.write('| --- | ' + ' | '.join(['---' for _ in range(len(competition_pivot.columns))]) + ' |\n')
    for idx, row in competition_pivot.iterrows():
        row_values = []
        for col in competition_pivot.columns:
            if col.startswith('环比变化'):
                row_values.append(f"{row[col]:.2%}")
            else:
                row_values.append(f"{row[col]:,.0f}")
        f.write(f"| {idx} | " + " | ".join(row_values) + " |\n")
    f.write('\n')
    
    # 星期几影响
    f.write('### 3.6 星期几影响\n\n')
    f.write('不同星期几的销售额变化：\n\n')
    f.write('![星期几销售额变化](figures/dow_sales.png)\n\n')
    
    # 特征重要性
    f.write('## 4. 特征重要性分析\n\n')
    f.write('通过随机森林模型分析各因素对销售额的重要性：\n\n')
    f.write('| 特征 | 重要性 |\n')
    f.write('| --- | --- |\n')
    for _, row in feature_importance.iterrows():
        f.write(f"| {row['Feature']} | {row['Importance']:.4f} |\n")
    f.write('\n![特征重要性](figures/feature_importance.png)\n\n')
    
    # 贡献度分析
    f.write('## 5. 各因素对环比变化的贡献度\n\n')
    if not contribution_df.empty:
        f.write('| 因素 | 类别 | 上月占比 | 当月占比 | 占比变化 | 贡献度 |\n')
        f.write('| --- | --- | --- | --- | --- | --- |\n')
        for _, row in contribution_df.head(15).iterrows():
            f.write(f"| {row['因素']} | {row['类别']} | {row['上月占比']:.2%} | {row['当月占比']:.2%} | {row['占比变化']:.2%} | {row['贡献度']:.2%} |\n")
        f.write('\n![贡献度瀑布图](figures/contribution_waterfall.png)\n\n')
    
    # 结论
    f.write('## 6. 分析结论\n\n')
    f.write('根据上述分析，我们得出以下结论：\n\n')
    
    # 这里会在脚本执行后根据实际分析结果手动补充
    f.write('1. 最近三个月销售额环比变化显著，主要影响因素为...\n')
    f.write('2. 从店铺类型来看，X类型店铺表现最好，环比增长...\n')
    f.write('3. 促销活动对销售额有明显影响，实施促销的店铺环比增长...\n')
    f.write('4. 商品种类方面，X种类商品的销售额贡献最大...\n')
    f.write('5. 竞争对手距离对销售额的影响表现为...\n')
    f.write('6. 随机森林模型分析显示，最重要的特征是...\n\n')
    
    f.write('## 7. 建议措施\n\n')
    f.write('基于上述分析，我们提出以下建议：\n\n')
    f.write('1. 针对表现最好的店铺类型，可以...\n')
    f.write('2. 优化促销策略，重点关注...\n')
    f.write('3. 调整商品结构，增加...\n')
    f.write('4. 考虑竞争环境因素，在...\n')

print("\n分析完成！结果已保存到results目录下。") 