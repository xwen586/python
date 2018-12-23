#!/usr/bin/python
#-*- coding:UTF-8 -*-
# 网络编程（使用线程技术 Threading）
#a03thrdsvr.py
from socketserver import TCPServer, ThreadingMixIn, StreamRequestHandler

class Server(ThreadingMixIn, TCPServer): pass

class Handler(StreamRequestHandler):
	def handle(self):
		addr = self.request.getpeername()
		print('Got connection from:', addr)
		self.wfile.write(bytes('Thank you for connecting (Threading)', encoding="utf-8"))

#server = TCPServer(('', 1234), Handler)
server = TCPServer(("127.0.0.1", 8080), Handler)
server.serve_forever()
