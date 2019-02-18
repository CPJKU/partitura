#!/usr/bin/env python

from functools import wraps
from collections import defaultdict

def cached_property(func, name=None):
    """
    cached_property(func, name=None) -> a descriptor
    This decorator implements an object's property which is computed
    the first time it is accessed, and which value is then stored in
    the object's __dict__ for later use. If the attribute is deleted,
    the value will be recomputed the next time it is accessed.

    Usage:

    >>> class X(object):
    >>>     @cached_property
    >>>     def foo(self):
    >>>         return slow_computation()
    >>> x = X()
    >>> # first access of foo(slow):
    >>> x.foo
    >>> # subsequent access of foo (fast):
    >>> x.foo
    """
    if name is None :
        name = func.__name__

    @wraps(func)
    def _get(self):
        try :
            return self.__dict__[name]
        except KeyError :
            self.__dict__[name] = func(self)
            return self.__dict__[name]

    @wraps(func)
    def _set(self, value):
        self.__dict__[name] = value

    @wraps(func)
    def _del(self ):
        self.__dict__.pop(name, None)

    return property(_get, _set, _del)


def iter_subclasses(cls, _seen = None):
    """
    iter_subclasses(cls)

    Generator over all subclasses of a given class, in depth first order.

    >>> list(iter_subclasses(int)) == [bool]
    True
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>> 
    >>> for cls in iter_subclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
    >>> # get ALL (new-style) classes currently defined
    >>> [cls.__name__ for cls in iter_subclasses(object)] #doctest: +ELLIPSIS
    ['type', ...'tuple', ...]
    """
    
    if not isinstance(cls, type):
        raise TypeError('iter_subclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None: _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError: # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in iter_subclasses(sub, _seen):
                yield sub


class ComparableMixin(object):
    """source:
    http://regebro.wordpress.com/2010/12/13/python-implementing-rich-comparison-the-correct-way/
    """
    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            # _cmpkey not implemented, or return different type,
            # so I can't compare with "other".
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s,o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s,o: s <= o)

    def __eq__(self, other):
       return self._compare(other, lambda s,o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s,o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s,o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s,o: s != o)


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
