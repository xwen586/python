#-*- coding:utf-8 -*-
# 列表List

list('Hello')

# 列表操作
x = [1, 1, 1]
x[1] = 2
print( x )

# 分片修改
name = list('Perl')
name[2:] = list('ython')
print( name )

# 列表元素增、删、改
names = ['Alice', 'Beth', 'Cecil', 'Dee-Dee', 'Earl']
names.append('Tomcat')  # 尾部追加
print( names )
del names[2]
names.remove('Tomcat')
print( names )

# 列表复制
names = ['Alice', 'Beth', 'Cecil', 'Dee-Dee', 'Earl']
n = names[:]   #通过切片副本来复制
n[0] = 'Mr. Gumby'  # Change the list，不会改变names[0] 的值
n1 = names
n1[0]= 'Mr. Gumby'  # 改变了names[0] 的值

#列表方法
lst = [1, 2, 3]
lst.append(5)
print(lst)
lst.append(5)
lst.count(5)  # 统计某元素在列表中的数量
lst.clear()

# 扩展
a = [1, 2, 3]
b = [4, 5, 6]
a.extend(b)    # 等同 a=a+b

