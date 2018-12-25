#!/usr/bin/env python
#-*- coding:UTF-8 -*-
# python 安装包 , 使用py2exe

'''
# http://www.py2exe.org/index.cgi/Tutorial
命令：python setup2.py py2exe

使用Python3.6及以上，报错  IndexError: tuple index out of range
py2exe支持的最高版本为py3.4
解决：使用pyinstaller（3.3.1）打包，打包py3.6版本的程序没有问题。
'''
from distutils.core import setup
import py2exe

# setup(console=['hello.py'])
setup(windows=['hello.py'])

