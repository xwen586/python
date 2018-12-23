#!/usr/bin/python
#-*- coding:UTF-8 -*-
# GUI 编程(wxPython)

import wx
app = wx.App()
w = wx.Frame(None, title='Hello')
btn = wx.Button(w, label='Click')
w.show()
app.MainLoop()

