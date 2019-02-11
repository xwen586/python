#!/usr/bin/env python3
# a2MeanReducer.py
"""分布式计算均值和方差的reducer
原 mrMeanReducer.py
1）先确认输入文件 inputFile.txt
2）执行命令：
Linux>inputFile.txt | python a1MeanMapper.py | python a2MeanReducer.py
Dos>python a1MeanMapper.py < .\data\inputFile.txt | python a2MeanReducer.py
"""
import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()
       
input = read_input(sys.stdin)#creates a list of input lines

#split input lines into separate items and store in list of lists
mapperOut = [line.split('\t') for line in input]

#accumulate total number of samples, overall sum and overall sum sq
cumVal=0.0
cumSumSq=0.0
cumN=0.0
for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    cumVal += nj*float(instance[1])
    cumSumSq += nj*float(instance[2])
    
#calculate means
mean = cumVal/cumN
meanSq = cumSumSq/cumN

#output size, mean, mean(square values)
print("%d\t%f\t%f" % (cumN, mean, meanSq))
#print >> sys.stderr, "report: still alive"
print("report: still alive", file=sys.stderr)
