#-*- coding:utf-8 -*-
# 类


class a01class(object):
    """description of class"""

class Person:
    def set_name(self, name):
        self.name = name
    def get_name(self):
        return self.name
    def greet(self):
        print("Hello, world! I'm {}.".format(self.name))

foo = Person()
foo.set_name('Luke Skywalker')
foo.greet()
