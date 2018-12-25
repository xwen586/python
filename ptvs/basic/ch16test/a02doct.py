#!/usr/bin/env python
#-*- coding:UTF-8 -*-
# a02doct.py
# doctest 文档测试工具
# python -m doctest a02doct.py -v

def square(x):
    '''Squares a number and returns the result.
    >>> square(2)
    4
    >>> square(3)
    9
    '''
    return x * x

def myfunc(a, b):
    """
    >>> myfunc(2, 3)
    6
    >>> myfunc('a', 3)
    'aaa'
    """
    return a * b

# 定义了main后，命令行简化为：python a02doct.py -v
if __name__ =='__main__':
    import doctest, a02doct
    doctest.testmod(a02doct)
