#!/usr/bin/env python
#-*- coding:UTF-8 -*-
# a06cgi2.py
# 利用FieldStorage 获取一个值的CGI脚本
# 访问：http://localhost:8001/htbin/a06cgi2.py

import cgi

form = cgi.FieldStorage() 
name = form.getvalue('name', 'CGI world2')
print('Content-type: text/plain\n')
print('Hello, {}!'.format(name))
