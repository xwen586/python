#!/usr/bin/env python3
# a02polt.py
"""
a01check.py 中完成数据访问与下载
本程序展示 Matplotlib 库图功能
"""
# 读取数据，并添加标题行：
# sepal length/width-花萼长/宽
# petal length/width-花瓣长/宽
import pandas as pd
df = pd.read_csv(r'.\ch01prep\iris.data', 
                 names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

''' ---------- 可视化 ---------- 
Matplotlib 库，Python 绘图库的鼻祖
'''
#内嵌画图
#%matplotlib inline  # 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl  # 解决中文字体问题
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

plt.style.use('ggplot')  # 风格设置为近似R 中的ggplot 库（这需要matplotlib 1.41）

'''1、图的样例'''
fig, ax = plt.subplots(figsize=(6,4))   # 创建宽度为6 英寸和高度为4 英寸的一个插图。
ax.hist(df['petal width'], color='black')  # 调用.hist()并传入数据
ax.set_ylabel('Count', fontsize=12)  # 设置座标
ax.set_xlabel('Width', fontsize=12)
plt.title('Iris Petal Width', fontsize=14, y=1.01)  # 设置标题：鸢尾花花瓣宽度
plt.show()

'''2、为iris 数据集的每一列生成直方图'''
fig, ax = plt.subplots(2,2, figsize=(6,4))
# 花瓣宽度
ax[0][0].hist(df['petal width'], color='black');
ax[0][0].set_ylabel('Count', fontsize=12)
ax[0][0].set_xlabel('Width', fontsize=12)
ax[0][0].set_title('花瓣宽 Iris Petal Width', fontsize=14, y=1.01)
# 花瓣长度
ax[0][1].hist(df['petal length'], color='black');
ax[0][1].set_ylabel('Count', fontsize=12)
ax[0][1].set_xlabel('Lenth', fontsize=12)
ax[0][1].set_title('花瓣长 Iris Petal Lenth', fontsize=14, y=1.01)
# 花萼宽度
ax[1][0].hist(df['sepal width'], color='black');
ax[1][0].set_ylabel('Count', fontsize=12)
ax[1][0].set_xlabel('Width', fontsize=12)
ax[1][0].set_title('花萼宽 Iris Sepal Width', fontsize=14, y=1.01)
# 花萼长度
ax[1][1].hist(df['sepal length'], color='black');
ax[1][1].set_ylabel('Count', fontsize=12)
ax[1][1].set_xlabel('Length', fontsize=12)
ax[1][1].set_title('花萼长 Iris Sepal Length', fontsize=14, y=1.01)
plt.tight_layout()  #自动调整子插图
plt.show()

'''3、为iris 数据集生成散点图'''
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(df['petal width'], df['petal length'], color='green')
ax.set_xlabel('花瓣宽 Petal Width')
ax.set_ylabel('花瓣长 Petal Length')
ax.set_title('花瓣散点图 Petal Scatterplot')
plt.show()

'''4、为iris 线图'''
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(df['petal length'], color='blue')
ax.set_xlabel('Specimen Number')  
ax.set_ylabel('Petal Length')
ax.set_title('Petal Length Plot')
plt.show()

'''5、为iris 条形图'''
fig, ax = plt.subplots(figsize=(6,6))
bar_width = .8  
labels = [x for x in df.columns if 'length' in x or 'width' in x]
ver_y = [df[df['class']=='Iris-versicolor'][x].mean() for x in labels]
vir_y = [df[df['class']=='Iris-virginica'][x].mean() for x in labels]
set_y = [df[df['class']=='Iris-setosa'][x].mean() for x in labels]
x = np.arange(len(labels))
ax.bar(x, vir_y, bar_width, bottom=set_y, color='darkgrey')
ax.bar(x, set_y, bar_width, bottom=ver_y, color='white')
ax.bar(x, ver_y, bar_width, color='black')
ax.set_xticks(x + (bar_width/2))
ax.set_xticklabels(labels, rotation=-70, fontsize=12);
ax.set_title('Mean Feature Measurement By Class', y=1.01)
ax.legend(['Virginica','Setosa','Versicolor'])
plt.show()

