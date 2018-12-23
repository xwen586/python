#!/usr/bin/python
#-*- coding:UTF-8 -*-
# 远程连接（）

from urllib.request import urlopen
import re

# 打开远程文件
webpage = urlopen('http://www.python.org')
firstLine = webpage.readline()   #读取html页面的第一行
print( firstLine )
text = webpage.read()  # 读取全文
print( text )

# 获取远程文件
urlretrieve('http://www.python.org', r'.\python_webpage.html')
