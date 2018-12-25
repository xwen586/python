#!/usr/bin/python
#-*- coding:UTF-8 -*-
# 网络编程（使用异步技术 poll函数）未调试
#a03pollsvr.py
import socket, select

s = socket.socket()
host = socket.gethostname()
port = 1234
# s.bind((host, port))
s.bind(("127.0.0.1", 8080))

fdmap = {s.fileno() : s}
s.listen(5)
p = select.poll()
p.register(s)
while True:
    events = p.poll()
    for fd, event in events:
        if fd in fdmap:
            c, addr = s.accept()
            print('Got connection from', addr)
            p.register(c)
            fdmap[c.fileno()] = c
        elif event & select.POLLIN:
            data = fdmap[fd].recv(1024)
            if not data: # No data -- connection closed
                print(fdmap[fd].getpeername(), 'disconnected')
                p.unregister(fd)
                del fdmap[fd]
            else:
                print(data)
