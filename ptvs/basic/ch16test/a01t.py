#!/usr/bin/env python
#-*- coding:UTF-8 -*-
# A Simple Test Program 原始的测试程序
# from area import rect_area

height = 3
width = 4
correct_answer = 12
answer = rect_area(height, width)
if answer == correct_answer:
    print('Test passed ')
else:
    print('Test failed ')

def rect_area(height, width):
    return height * height # This is wrong ...
