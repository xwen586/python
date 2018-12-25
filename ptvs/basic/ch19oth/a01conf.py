#!/usr/bin/env python
#-*- coding:UTF-8 -*-
# a01conf.py  配置
import os
from configparser import ConfigParser

print("当前目录："+ os.getcwd())

''' 打开当前目录的命令窗口，执行：
python a01conf.py
'''

CONFIGFILE = r".\area.ini"
config = ConfigParser()
# Read the configuration file:
config.read(CONFIGFILE)

# Print out an initial greeting;
# 'messages' is the section to look in:
print(config['messages'].get('greeting'))

# Read in the radius, using a question from the config file:
radius = float(input(config['messages'].get('question') + ' '))
# Print a result message from the config file;
# end with a space to stay on same line:
print(config['messages'].get('result_message'), end=' ')
# getfloat() converts the config value to a float:
print(config['numbers'].getfloat('pi') * radius**2)

items = config.items('DEFAULT')
print("Items:" )
print( items )
