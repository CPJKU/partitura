#!/usr/bin/env python

"""
Common container-related functionality
"""

from collections import defaultdict, OrderedDict, Callable

# taken from:
# http://stackoverflow.com/questions/6190331/can-i-do-an-ordered-default-dict-in-python

class DefaultOrderedDict(OrderedDict):
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
            not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, list(self.items())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(list(self.items())))
    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                        OrderedDict.__repr__(self))

def merge_dicts(a, b, _path=None):
    """
    Recursively merge dictionary b into a. 
    Values in b override values in a.

    """

    if _path is None: _path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], _path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                a[key] = b[key]
                #raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def partition(func, iterable):
    """
    Return a dictionary containing the equivalence classes (actually bags)
    of iterable, partioned according to func. The value of a key k is the 
    list of all elements e from iterable such that k = func(e)
    """
    result = defaultdict(list)
    for v in iterable:
        result[func(v)].append(v) 
    return result

def argpartition(func, iterable):
    """
    Return a dictionary containing the equivalence classes (actually bags)
    of iterable, partioned according to func. The value of a key k is the 
    list of all integers i such that k = func(iterable[i])
    """
    result = defaultdict(list)
    for i,v in enumerate(iterable):
        result[func(v)].append(i) 
    return result

def merge_dicts(a, b, path=None):
    """
    Merge dictionary b into dictionary a recursively. 
    In case of coinciding (leaf) values in a and b,
    values in b are returned
    """
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                a[key] = b[key]
                #raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

if __name__ == '__main__':
    pass
