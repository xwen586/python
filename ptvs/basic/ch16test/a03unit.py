#!/usr/bin/env python
#-*- coding:UTF-8 -*-
# a03unit.py
# unittest 单元测试工具

import unittest
'''几个概念：
TestCase 也就是测试用例
TestSuite 多个测试用例集合在一起，就是TestSuite
TestLoader是用来加载TestCase到TestSuite中的
TestRunner是来执行测试用例的,测试的结果会保存到TestResult实例中，包括运行了多少测试用例，成功了多少，失败了多少等信息
'''

# ------ 第1个实例 ------
def  cal(a,b):
    if a == 7 and b == 9:
        return 'An insidious bug has surfaced!'
    else:
        return a+b

class CalTest(unittest.TestCase):
    def testA(self):
        expected = 6
        result = cal(2, 4)
        self.assertEqual(expected, result)
        
    def testB(self):
        expected = 16
        result = cal(7, 9)
        self.assertEqual(expected, result, "testB() have bug！")


# ------ 第2个实例 ------
def product(x, y):
    return x * y   # 执行python a03unit.py 后，Ran 2 tests in 0.002s
    #pass   # 执行python a03unit.py 后，显示错误信息。

class ProductTestCase(unittest.TestCase):  # 继承unittest.TestCase
    def test_integers(self):
        for x in range(-10, 10):
            for y in range(-10, 10):
                p = product(x, y)
                self.assertEqual(p, x * y, 'Integer multiplication failed')
    def test_floats(self):
        for x in range(-10, 10):
            for y in range(-10, 10):
                x = x / 10
                y = y / 10
                p = product(x, y)
                self.assertEqual(p, x * y, 'Float multiplication failed')


if __name__ == '__main__': 
    unittest.main()  # 负责运行测试，实例化所有TestCase子类，运行test开头的方法
