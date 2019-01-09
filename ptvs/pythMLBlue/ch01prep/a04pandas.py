#!/usr/bin/env python3
# a04pandas.py
"""
了解pandas 的Series.map()、Series.apply()、DataFrame.apply()、
DataFrame.applymap()和 DataFrame.groupby()方法
"""
import pandas as pd
import numpy as np
df = pd.read_csv(r'.\ch01prep\iris.data', 
                 names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

'''--------------
1．Map 方法适用于序列数据
用map方法将class列值进行替换。
'''
df['class'] = df['class'].map({'Iris-setosa':'SET', 'Iris-virginica':'VIR', 'Iris-versicolor':'VER'})

'''--------------
2、Apply方法
新增一列“wide petal”宽花瓣，判断 petal width 值，大于1.3编码为1，否则为0
'''
df['wide petal'] = df['petal width'].apply(lambda v: 1 if v >= 1.3 else 0)
# 花瓣面积。对df 操作，axis=1表示行操作，axis=0表示列操作
df['petal area'] = df.apply(lambda r: r['petal length'] * r['petal width'], axis=1)

'''--------------
3．Applymap 数据单元操作
'''
# 如果某个值是float 类型，则进行log计算
df.applymap(lambda v: np.log(v) if isinstance(v, float) else v)

'''--------------
4．Groupby 分组
'''
df.groupby('class').mean()  # 按class分组，并求均值
df.groupby('class').describe()  # 按class分组，并描述性统计
df.groupby('petal width')['class'].unique().to_frame()

df.groupby('class')['petal width'].agg({'delta': lambda x: x.max() - x.min(), 'max': np.max, 'min': np.min})
