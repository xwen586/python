#!/usr/bin/python
#-*- coding:UTF-8 -*-
# a01rpt.py
'''需安装
pip install reportlab
'''
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics import renderPDF

d = Drawing(100, 100)
s = String(50, 50, 'Hello, world!', textAnchor='middle')
d.add(s)
renderPDF.drawToFile(d, 'hello.pdf', 'A simple PDF file')

print( "pdf create success!")
