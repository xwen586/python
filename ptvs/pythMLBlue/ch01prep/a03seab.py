#!/usr/bin/env python3
# a03seab.py
"""
a01check.py 中完成数据访问与下载
本程序展示 Seaborn 库图功能
"""
# 读取数据
import pandas as pd
df = pd.read_csv(r'.\ch01prep\iris.data', 
                 names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

''' ---------- 可视化 ---------- 
Seaborn 库，专为统计可视化而创建的库
'''
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue="class")

fig, ax = plt.subplots(2, 2, figsize=(7, 7))
sns.set(style='white', palette='muted')
sns.violinplot(x=df['class'], y=df['sepal length'], ax=ax[0,0])
sns.violinplot(x=df['class'], y=df['sepal width'], ax=ax[0,1])
sns.violinplot(x=df['class'], y=df['petal length'], ax=ax[1,0])
sns.violinplot(x=df['class'], y=df['petal width'], ax=ax[1,1])
fig.suptitle('Violin Plots', fontsize=16, y=1.03)

for i in ax.flat:
    plt.setp(i.get_xticklabels(), rotation=-90)
fig.tight_layout()

plt.show()
