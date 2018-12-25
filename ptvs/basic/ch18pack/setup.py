#!/usr/bin/env python
#-*- coding:UTF-8 -*-
# python 安装包
'''
命令行：
python setup.py build     #编译
python setup.py install    #安装
python setup.py sdist      #制作分发包
python setup.py bdist_wininst    #制作windows下的分发包
'''
from setuptools import setup

setup(
    name = "mydemo",
    version = "0.1",
    description='A simple example',
    author='Magnus Lie Hetland',
    py_modules=['hello']
)


