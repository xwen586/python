#!/usr/bin/python
#-*- coding:UTF-8 -*-
#a03soup.py
# Web编程: 屏幕抓取。使用BeautifulSoup
# 安装： $ pip install beautifulsoup4
from urllib.request import urlopen
from bs4 import BeautifulSoup

text = urlopen('http://python.org/jobs').read()
soup = BeautifulSoup(text, 'html.parser')
jobs = set()
for job in soup.body.section('h2'):
    jobs.add('{} ({})'.format(job.a.string, job.a['href']))
print('\n'.join(sorted(jobs, key=str.lower)))
