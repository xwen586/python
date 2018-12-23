#-*- coding:utf-8 -*-
# 编码

print(  "Hello, world!".encode("ASCII") )
print(  "Hello, world!".encode("UTF-8") )
print(  "Hello, world!".encode("UTF-32") )

len("Hello？".encode("UTF-8"))   # =8
