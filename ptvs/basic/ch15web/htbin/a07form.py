#!/usr/bin/env python
#-*- coding:UTF-8 -*-
# a07form.py
# 页面表单 
# 访问：http://localhost:8001/htbin/a07form.py

import cgi
form = cgi.FieldStorage()
name = form.getvalue('name', 'CGI world3')

print("Content-type: text/html\n")
print("""<html>
<head>
<title>Greeting Page</title>
</head>
<body>
  <h1>Hello, {}!</h1>
  <form action='a07form.py'>
    Change name
    <input type='text' name='name' />
    <input type='submit' />
  </form>
</body>
</html>
""".format(name))
