#!/usr/bin/python
#-*- coding:UTF-8 -*-
# a04cgi.py
# A Simple CGI Script, 执行py脚本文件，开启cgi映射
# python -m http.server --cgi 8001
# 本文件放在上边命令所在目录下的 /cgi-bin 或 /htbin 目录中，进行CGI脚本解析
# 访问：http://localhost:8001/htbin/a04cgi.py

print('Content-type: text/plain')
print() # Prints an empty line, to end the headers
print('Hello, world!')
