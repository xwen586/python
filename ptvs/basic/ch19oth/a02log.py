#!/usr/bin/env python
#-*- coding:UTF-8 -*-
# a02log.py  日志
import logging

logging.basicConfig(level=logging.INFO, filename='mylog.log')

logging.info('Starting program')
logging.info('Trying to divide 1 by 0')
print(1 / 1)
logging.info('The division succeeded')
logging.info('Ending program')
