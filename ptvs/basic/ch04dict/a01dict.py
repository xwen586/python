#-*- coding:utf-8 -*-
# 字典

# 字典是由 {} 和 值名对组成
phonebook = {'Alice': '2341', 'Beth': '9102', 'Cecil': '3258'}

# dict函数，生成转换成字典方法
items = [('name', 'Gumby'), ('age', 42)]
d1 = dict(items)
print(d1)
d2 = dict(name='Gumby', age=42)
print(d2)

# 字典结构嵌套
person={}
person["name"]={}
person["age"]={}
person["tel"]={}
person["tel"]["phone"]={}
person["tel"]["home"]={}

people = {
	'Alice': { 'phone': '2341', 	'addr': 'Foo drive 23'},
	'Beth': { 'phone': '9102',  'addr': 'Bar street 42'},
	'Cecil': { 'phone': '3158',  'addr': 'Baz avenue 90'}
}
