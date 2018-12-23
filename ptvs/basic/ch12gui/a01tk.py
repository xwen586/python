#!/usr/bin/python
#-*- coding:UTF-8 -*-
# GUI 编程(Tkinter)
# Tkinter 模块(Tk 接口)是 Python 的标准 Tk GUI 工具包的接口

import tkinter as tk

top = tk.Tk()
# 进入消息循环
top.mainloop()


def clicked():
    print('I was clicked!')

root = tk.Tk()
li     = ['C','python','php','html','SQL','java']
movie  = ['CSS','jQuery','Bootstrap']
listb  = tk.Listbox(root)          #  创建两个列表组件
listb2 = tk.Listbox(root)
btn = tk.Button()
for item in li:                 # 第一个小部件插入数据
    listb.insert(0,item)
 
for item in movie:              # 第二个小部件插入数据
    listb2.insert(0,item)

btn['text'] = 'Click me!'
btn['command'] = clicked

btn.pack()
listb.pack()                    # 将小部件放置到主窗口中
listb2.pack()

root.mainloop()                 # 进入消息循环
