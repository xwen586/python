#!/usr/bin/python
#-*- coding:UTF-8 -*-
#a01scrap.py
# Web编程: 屏幕抓取，是程序下载网页并提取信息的过程
# A Simple Screen-Scraping Program
from urllib.request import urlopen

import re
p = re.compile('<a href="(/jobs/\\d+)/">(.*?)</a>')
text = urlopen('http://python.org/jobs').read().decode()

for url, name in p.findall(text):
    print('{} ({})'.format(name, url))
