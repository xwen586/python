#!/usr/bin/env python
#!/usr/bin/python
#-*- coding:UTF-8 -*-
# a04faulty.py
# 使用cgitb 调试，调用回溯的CGI脚本
# python -m http.server --cgi 8001
# 本文件放在上边命令所在目录下的 /cgi-bin 或 /htbin 目录中，进行CGI脚本解析
# 访问：http://localhost:8001/htbin/a05faulty.py
import cgitb; cgitb.enable()

print('Content-type: text/html\n')
print(1/0)
print('Hello, world!')
