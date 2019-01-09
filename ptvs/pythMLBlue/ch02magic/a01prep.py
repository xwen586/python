#!/usr/bin/env python3
# a01prep.py
"""
description of class
"""
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
#%matplotlib inline

pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.precision", 3)

# Use the file location of your Import.io csv
CSV_PATH = r"./ch02magic/magic.csv"
df = pd.read_csv(CSV_PATH)
df.columns  # 显示列标题
len(df)  # 记录总数
df.T    #转置 查看数据的某些样本。有一些缺失值（NaN）
df.head().T  #前5条数据
df.T.ix[:,1:2]  #  索引第2、3条数据

# 检索 'routable_link/_text' 为 '203 Rivington' 或 '280 E 2nd' 的数据项。
nn = df[df['routable_link/_text'].str.contains('203 Rivington')|df['routable_link/_text'].str.contains('280 E 2nd')]
nn[:2].T  # 显示最后2条

# 两种类型的房源：
#一种类型是单个单元 Apartment for Rent，一种类型是多个单元Apartments for Rent。
df['listingtype_value']  # 查看房源类型
# 将数据拆分为单一的单元、多个单元
# multiple units
mu = df[ df['listingtype_value'].str.contains('Apartments For') ]
len(mu)
# single units
su = df[df['listingtype_value'].str.contains('Apartment For')]
len(su)

# 检查“价格列”中是否有空值（单一单元的）
len( su[su['pricelarge_value_prices'].isnull()] )

# 将数据格式化为标准结构。例如，至少需要为卧室数、浴室数、平方英尺和地址各准备一列。
su['propertyinfo_value']
#检查没有包含'bd'或'Studio'的行数
len(su[~(su['propertyinfo_value'].str.contains('Studio') | \
    su['propertyinfo_value'].str.contains('bd'))])
#检查没有包含'ba'的行数
len(su[~(su['propertyinfo_value'].str.contains('ba'))])

# 对缺失数据处理
# 没有浴室的房源
no_baths = su[~(su['propertyinfo_value'].str.contains('ba'))]
# 再排除那些缺失了浴室信息的房源
sucln = su[~su.index.isin(no_baths.index)]
# 继续解析卧室和浴室信息：
# 使用项目符号进行切分
def parse_info(row):
        if not 'sqft' in row:
            br, ba = row.split('•')[:2]
            sqft = np.nan
        else:
            br, ba, sqft = row.split('•')[:3]                
        return pd.Series({'Beds': br, 'Baths': ba, 'Sqft': sqft})
#
attr = sucln['propertyinfo_value'].apply(parse_info)
# 在取值中将字符串删除，如：bd、ba 和sqft，
attr_cln = attr.applymap(lambda x: x.strip().split(' ')[0] if isinstance(x,str) else np.nan)
attr_cln

# 将上一计算数据拼接到sucln表中
sujnd = sucln.join(attr_cln)

'''查看 routable_link/_text 字段信息，含有楼层floor，邮编zip 信息'''
sujnd['routable_link/_text'] 
# 定义解析函数 parse out zip, floor
def parse_addy(r):
    so_zip = re.search(', NY(\d+)', r)
    so_flr = re.search('(?:APT|#)\s+(\d+)[A-Z]+,', r)
    if so_zip:
        zipc = so_zip.group(1)
    else:
        zipc = np.nan
    if so_flr:
        flr = so_flr.group(1)
    else:
        flr = np.nan
    return pd.Series({'Zip':zipc, 'Floor': flr})
# 从 'routable_link/_text' 中解析 Floor、Zip
flrzip = sujnd['routable_link/_text'].apply(parse_addy)
print(flrzip)
len(flrzip[~flrzip['Floor'].isnull()]) # 楼层非空数量
len(flrzip[~flrzip['Zip'].isnull()])  # 带有邮编房源数量
# 楼层、邮编结果拼接到表中
suf = sujnd.join(flrzip)

# 将数据减少为所感兴趣的那些列
sudf = suf[['pricelarge_value_prices', 'Beds', 'Baths', 'Sqft', 'Floor', 'Zip']]
# 列名修改
sudf.rename(columns={'pricelarge_value_prices':'Rent'}, inplace=True)
# 重置索引
sudf.reset_index(drop=True, inplace=True)



''' ---------- 分析数据 ---------- '''
# 初步总体分析
sudf.describe()

# 工作室公寓认定为一个零卧室的公寓，将出现的'Studio'替换为0
sudf.loc[:,'Beds'] = sudf['Beds'].map(lambda x: 0 if 'Studio' in x else x)

# 统计数据的列必须是数值类型。从下一语句看，'Beds','Baths', 'Floor'并非都是
sudf.info()  # 'Beds','Baths', 'Floor' 都是object

# let's fix the datatype for the columns
sudf.loc[:,'Rent'] = sudf['Rent'].astype(int)  #置为整型
sudf.loc[:,'Beds'] = sudf['Beds'].astype(int)

# half baths require a float
sudf.loc[:, 'Baths'] = sudf['Baths'].astype(float)

# with NaNs we need float, but we have to replace commas first
sudf.loc[:, 'Sqft'] = sudf['Sqft'].str.replace(',', '') # 数字中的逗号去掉

sudf.loc[:, 'Sqft'] = sudf['Sqft'].astype(float)
sudf.loc[:, 'Floor'] = sudf['Floor'].astype(float)

sudf.info()  # 此时查看，应都修改为正确类型
sudf.describe()  #统计信息看，有个楼层为1107层

#找到1107层
suf[suf['Floor'].astype(float)>25].T
sudf[sudf['Floor'].astype(float)>25].T  #找到该记录，索引号为 318
# 删除索引318的记录
sudf = sudf.drop([318])

# 按照邮政编码来查看平均价格
sudf.pivot_table('Rent', 'Zip', 'Beds', aggfunc='mean')
# 基于房源的数量进行透视
sudf.pivot_table('Rent', 'Zip', 'Beds', aggfunc='count')



''' ---------- 可视化数据 ---------- 
需要安装组件：conda install -c conda-forge folium
访问 https://github.com/python-visualization/folium
'''
#由于缺少包含两到三间卧室的公寓，让我们缩减数据集，聚焦到工作室和一间卧室的房源。
su_lt_two = sudf[sudf['Beds']<2]

import folium
map = folium.Map(location=[40.748817, -73.985428], zoom_start=13)
# 此处无法进行，少nyc.json  E:\workspace\git\python\ptvs\pythMLBlue/ch02magic/nyc.json
map.geo_json(geo_path=r'/Users/alexcombs/Downloads/nyc.json', data=su_lt_two,
             columns=['Zip', 'Rent'],
             key_on='feature.properties.postalCode',
             threshold_scale=[1700.00, 1900.00, 2100.00, 2300.00, 2500.00, 2750.00],
             fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
             legend_name='Rent (%)',
              reset=True)
map.create_map(path='nyc.html')

su_lt_two.sort('Zip')

#---------数据建模----------
import patsy
import statsmodels.api as sm

f = 'Rent ~ Zip + Beds'
y, X = patsy.dmatrices(f, su_lt_two, return_type='dataframe')

results = sm.OLS(y, X).fit()
print(results.summary())

results.params

to_pred_idx = X.iloc[0].index
to_pred_zeros = np.zeros(len(to_pred_idx))
tpdf = pd.DataFrame(to_pred_zeros, index=to_pred_idx, columns=['value'])
print(tpdf)

tpdf['value'] = 0
tpdf.loc['Intercept'] = 1
tpdf.loc['Beds'] = 2
tpdf.loc['Zip[T.10002]'] = 1
print(tpdf)
