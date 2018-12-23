#!/usr/bin/python
#-*- coding:UTF-8 -*-
# 网络编程（使用分叉技术 Forking，注：windows不支持分叉）
#a03forksvr.py
from socketserver import TCPServer, ForkingMixIn, StreamRequestHandler

class Server(ForkingMixIn, TCPServer): pass

class Handler(StreamRequestHandler):
	def handle(self):
		addr = self.request.getpeername()
		print('Got connection from:', addr)
		self.wfile.write(bytes('Thank you for connecting (Forking)', encoding="utf-8"))

#server = TCPServer(('', 1234), Handler)
server = TCPServer(("127.0.0.1", 8080), Handler)
server.serve_forever()
