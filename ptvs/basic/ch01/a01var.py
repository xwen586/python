#-*- coding:utf-8 -*-
# expert
# http://club.topsage.com/thread-361615-1-1.html

# 数字 与 计算
print( 1/2 )
print( 1%2 )     #取余 =1
print( 10 // 3 ) #整除 =3
print( -3 ** 2 )   # 幂运算
print( (-3) ** 2 )  

# 16进制，8进制
print( 0xAF ) # 16进制
print( 0o10 )  # 8进制： python2下 O10； 
print( 0b1011010010 )

#字符串
a = "hello world!a"
b = '你好！b'
print(a)
print(b)

name = "ada lovelace"
print(name.title())  # 首字母大写

print(repr("Hello,\nworld!"))
print( r'Let\'s go!')   #原始字符串

# 字符串拼接
s1 = "Let's say " '"Hello, world!"'  #书写字符串
s2 = "Hello, " "world！" + "python!"  # 拼接字符串
print(s1)
x = "Hello, "

x = 12
y = 3
#print("x * y = " + `x*y`)   #python3 中不使用反引号
print("x * y = " + repr(x*y))


#输入
input("The meaning of life: ")
x = input("x: ")
y = input("y: ")
print("x * y = " + str(int(x) * int(y)))
