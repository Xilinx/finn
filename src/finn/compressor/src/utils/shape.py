from typing import Tuple
from itertools import zip_longest

class Shape:
    def __init__(self, t: Tuple[int] = ()): self.t = tuple(t)

    def __len__(self): return len(self.t)

    def __iter__(self): return self.t.__iter__()
    
    def __getitem__(self, val):
        if type(val) == int and val >= len(self.t):
            return 0
        r = self.t.__getitem__(val)
        if type(r) == int:
            return r
        else:
            return Shape(r)
    
    def __lshift__(self, val):
        return Shape([0 for el in range(val)] + list(self.t))

    def __add__(self, val):
        return self.__binary_arithmetic_operation(val, lambda x,y: x+y)

    def __sub__(self, val):
        return self.__binary_arithmetic_operation(val, lambda x,y: x-y)

    def __binary_arithmetic_operation(self, val, op):
        if type(val) == int:
            return Shape([op(el, val) for el in self.t])
        elif type(val) == Shape:
            zipped = zip_longest(self.t, val.t, fillvalue=0)
            return Shape([op(a, b) for a, b in zipped])
        else:
            raise RuntimeError("Unsupported type.")
        
    def __repr__(self): return f"Shape {self.t[::-1]}"
    
    def __eq__(self, other):
        for col1, col2 in zip_longest(self, other, fillvalue=0):
            if col1 != col2: return False
        return True