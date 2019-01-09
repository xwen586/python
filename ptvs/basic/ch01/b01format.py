#-*- coding:utf-8 -*-
# b01format.py
"""格式化输出
"""

a = [1, 2, 3, 4]
b = ["first", "second", "third", "fourth"]
c = a + b
print("Output #3: {0}, {1}, {2}".format(a, b, c))

# 整数
x = 9
print("Output #4: {0:d}".format(x))
print("Output #5: {0}".format(3**4))
print("Output #6: {0}".format(int(8.3)/int(2.7)))

# 浮点数
print("Output #7: {0:.3f}".format(8.3/2.7))
y = 2.5*4.7
print("Output #8: {0:.1f}".format(y))
r = 8/float(3)
print("Output #9: {0:.2f}".format(r))
print("Output #10: {0:.4f}".format(8/3))

from math import exp, log, sqrt
print("Output #11: {0:.4f}".format(exp(3))) #e的乘方
print("Output #12: {0:.2f}".format(log(4))) #自然对数
print("Output #13: {0:.1f}".format(sqrt(81))) #平方根
