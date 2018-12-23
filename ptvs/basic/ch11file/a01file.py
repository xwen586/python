#-*- coding:utf-8 -*-
# 文件操作

f = open('somefile.txt', 'w')
f.write('Hello, World！')
f.close()

f = open('somefile.txt', 'r')
f.read(4)   # 输出 'Hell'
f.read()   # 输出'o, World!'

f = open(r'.\somefile.txt', 'a')
f.write('\n\n')
f.write('this is file append write！\n\n')
f.close()

f = open(r'.\somefile.txt')
for i in range(3):
    print(str(i) + ': ' + f.readline(), end='')

f = open(r'.\somefile.txt')
lines = f.readlines()
f.close()
lines.append('new line append\n\n')
lines[3] = "isn't a\n"
f = open(r'.\somefile.txt', 'w')
f.writelines(lines)
f.close()
