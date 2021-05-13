#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  :   naive dormin
# @time    :   2021/05/11 01:29:09
# @version :   1.0.0
import numpy as np

class Book(object):
    def __init__(self, title):
        self.title = title

    @classmethod
    def create(cls, title): 
        book = cls(title=title)
        return book


class Foo(object):
    """类三种方法语法形式"""

    def instance_method(self):
        print("是类{}的实例方法，只能被实例对象调用".format(Foo))

    @staticmethod
    def static_method():
        print("是静态方法")

    @classmethod
    def class_method(cls):
        print("是类方法")


class Fool(object):
    X = 1
    Y = 2

    @staticmethod
    def average(*mixes):
        return sum(mixes)/len(mixes)

    @staticmethod
    def static_method():
        return Fool.average(Fool.X, Fool.Y)

    @classmethod
    def class_method(cls):
        return cls.average(cls.X, cls.Y)


if __name__ == "__main__":
    fool = Fool()
    print(fool.static_method())
    print(fool.class_method())
