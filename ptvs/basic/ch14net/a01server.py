#!/usr/bin/python
#-*- coding:UTF-8 -*-
# 网络编程（socket）

import socket

s = socket.socket()
host = socket.gethostname()
port = 1234
#s.bind( (host, port) ) 
s.bind(("127.0.0.1",8080))
s.listen(5)
while True:
    c, addr = s.accept()
    print('Got connection from', addr)
    c.send(bytes('Thank you for connecting',encoding="utf-8"))
    c.close()
