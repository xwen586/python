#!/usr/bin/env python3
# a1MeanMapper.py
""" 分布式均值和方差计算的mapper
原 mrMeanMapper.py
1）先确认输入文件 inputFile.txt
2）执行命令：
Linux>cat inputFile.txt | python mrMeanMapper.py
Dos>python a1MeanMapper.py < .\data\inputFile.txt
"""
import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()
        
input = read_input(sys.stdin)#creates a list of input lines
input = [float(line) for line in input] #overwrite with floats
numInputs = len(input)
input = mat(input)
sqInput = power(input,2)

#output size, mean, mean(square values)
print("%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput))) #calc mean of columns
#print >> sys.stderr, "report: still alive" 
print("report: still alive", file=sys.stderr)
