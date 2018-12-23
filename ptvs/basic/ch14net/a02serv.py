#!/usr/bin/python
#-*- coding:UTF-8 -*-
# 网络编程（socket）
# Python3以后，SocketServer这个模块的命名变成了socketserver。
from socketserver import TCPServer, StreamRequestHandler

class Handler(StreamRequestHandler):
	def handle(self):
		addr = self.request.getpeername()
		print('Got connection from:', addr)
		#self.wfile.write('Thank you for connecting')
		self.wfile.write(bytes('Thank you for connecting', encoding="utf-8"))

#server = TCPServer(('', 1234), Handler)
server = TCPServer(("127.0.0.1", 8080), Handler)
server.serve_forever()
