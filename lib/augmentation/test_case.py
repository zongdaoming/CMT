# /*
#  * @Author: dorming 
#  * @Date: 2021-01-14 15:21:38 
#  * @Last Modified by:   dorming 
#  * @Last Modified time: 2021-01-14 15:21:38 
#  */
import numpy as np

class B(object):
    def __init__(self, *args, **kwargs):
        self.a = 1
        self.b = 2
        print(self)
        print("init", args, kwargs)
    def  __new__(cls, *args, **kwargs):
        print("new ", args, kwargs)
        test = super(B, cls).__new__(cls)
        print(test)
        return test

        
class Counter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@Counter
def foo():
    pass

for i in range(10):
    foo()

print(foo.count)  # 10