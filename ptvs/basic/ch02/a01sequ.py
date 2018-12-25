#-*- coding:utf-8 -*-
# 序列 sequence

edward = ['Edward Gumby', 42]
john = ['John Smith', 50]
database = [edward,  john]
print( database )

# 索引操作
g = 'Hello'
print( g[0] )

months = ['January', 'February', 'March',
'April', 'May', 'June', 'July', 'August', 'September',
'October', 'November', 'December'
]
print( months[6])


# 切片（Slicing）
n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n[ 3:6 ]    # =[4, 5, 6]
n[ :3]       # =[1, 2, 3]
n[-3:-1]   # =[8, 9]   倒数第一，不是10
n[-3:]      # =[8, 9, 10]

#步长
n[0:10:2]  # =[1, 3, 5, 7, 9]
n[::4]    # =[1, 5, 9]
n[::-2]   # =[10, 8, 6, 4, 2]
n[8::-2]   # =[9, 7, 5, 3, 1]

# 序列相加
a = [1, 2, 3] + [4, 5, 6]
# 相乘 Multiplication
p =  'python ' * 5
q = [36] * 10
s = []*10     # len(s) = 0
s = [None]*10   # len(s) = 10

# 判断成员
users = ['mlh', 'foo', 'bar']
input('Enter your user name: ') in users
s = 'Hello Python World!'
'thon' in s

# 长度，最大、最小
numbers = [100, 34, 678]
len(numbers)   # =3
max(numbers)  # =678
min(numbers)  # =34
max(2, 3)   # =3
min(9, 3, 2, 5)  # =2
