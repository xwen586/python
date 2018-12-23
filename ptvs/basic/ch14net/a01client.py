#!/usr/bin/python
#-*- coding:UTF-8 -*-
# 网络编程（socket）

import socket

s = socket.socket()
host = socket.gethostname()
port = 1234
s.connect(("127.0.0.1", 8080))  #(host, port)
print(s.recv(1024))
