#!/usr/bin/env python3
# a1trgui.py
from numpy   import *
from tkinter import * #python3
import a1cart as regTrees  #引入构建的算法

import matplotlib
matplotlib.use('TkAgg') #设置后端TkAgg，将绘图窗口调到最前面。

#将TkAgg和matplotlib链接起来
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#class a1trgui(object):
"""构建GUI展示类"""

# 重画方法
def reDraw(tolS, tolN):
    reDraw.f.clf()  #清空之前的图像
    reDraw.a = reDraw.f.add_subplot(111)#重新添加新图
    if chkBtnVar.get(): #检查选框model tree是否被选中
        if tolN < 2: tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, regTrees.modelTreeEval)
    else:
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    #绘制真实值
    #reDraw.a.scatter(reDraw.rawDat[:, 0], reDraw.rawDat[:, 1], s=5)  # 绘制真实值，Python3有误
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)
    #绘制预测值
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # 绘制预测值
    reDraw.canvas.show()

def getInputs():#获取输入
    try:#期望输入是整数
        tolN = int(tolNentry.get())
    except:#清楚错误用默认值替换
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:#期望输入是浮点数
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS

def drawNewTree():
    tolN, tolS = getInputs()  # 从输入文本框中获取参数
    reDraw(tolS, tolN)  #绘制图

#def Run():
print("Hello class!")
root=Tk()
# 创建画布
reDraw.f = Figure(figsize=(5,4), dpi=100) #create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0,'10')
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0,'1.0')
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(regTrees.loadDataSet(r'./ch09tree/data/sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)
root.mainloop()
    

#if __name__ == '__main__':
#    #a = a1trgui()
#    Run()
#    #del a

