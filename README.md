# Rossmann销售数据环比归因分析项目

## 项目简介

本项目基于Rossmann连锁药店的销售数据，通过多维度归因分析，找出最近三个月（2015年5月、6月、7月）营业额环比变化的主要原因及各维度/因素的贡献度。

## 项目结构

```
.
├── README.md
├── visualization.py          # 数据可视化脚本
├── simple_visualization.py   # 简化版可视化脚本
├── results/                  # 分析结果目录
│   ├── figures/             # 图表文件
│   ├── summary.md           # 项目总结
│   ├── analysis_report.md   # 详细分析报告
│   └── attribution_model.md # 归因模型说明
└── datasets/                 # 数据文件（不包含在GitHub中）
```

## 主要功能

1. 数据探索与预处理
2. 月度销售趋势分析
3. 环比变化归因分析
4. 多维度因素贡献度计算
5. 可视化展示
6. 业务建议生成

## 使用方法

1. 安装依赖：
```bash
pip install pandas numpy matplotlib seaborn
```

2. 运行分析：
```bash
python visualization.py
```

3. 查看结果：
- 分析报告：`results/analysis_report.md`
- 项目总结：`results/summary.md`
- 归因模型：`results/attribution_model.md`

## 分析结果

主要发现：
- 6月销售额环比增长9.63%
- 促销活动和学校假期是主要增长因素
- 店铺类型和商品种类对增长有显著影响

详细分析结果请参考`results`目录下的文档。

## 技术栈

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License 