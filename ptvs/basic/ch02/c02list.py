#-*- coding:utf-8 -*-
# c02list.py
""" 列表用方括号[ ]来表示 """

bicycles = ['trek', 'cannondale', 'redline', 'specialized']
print(bicycles)
print(bicycles[0])  # 索引从0开始
print(bicycles[-1]) # 最后一个列表元素
message = "My first bicycle was a " + bicycles[0].title() + "." #首字母T大写
print(message)

motorcycles = ['honda', 'yamaha', 'suzuki']
print(motorcycles)
popped_motorcycle = motorcycles.pop()  # 末尾删除
print(motorcycles)
print(popped_motorcycle)

motorcycles.append("ducati")  # 末尾添加
motorcycles.sort()  # 排序
print(motorcycles)
motorcycles.remove('ducati')
len(motorcycles)
for motor in motorcycles:
   print(motor)
