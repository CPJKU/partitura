#!/usr/bin/env python

import logging
from functools import wraps
from collections import defaultdict

import numpy as np

LOGGER = logging.getLogger(__name__)

def iter_current_next(iterable):
    """
    Make an iterator that yields an (previous, current, next) tuple per element.

    Returns None if the value does not make sense (i.e. previous before
    first and next after last).

    Examples
    --------

    >>> l = []
    >>> list(iter_current_next(l))
    []
    >>> l = range(1)
    >>> list(iter_current_next(l))
    []
    >>> l = range(2)
    >>> list(iter_current_next(l))
    [(0, 1)]
    >>> l = range(3)
    >>> list(iter_current_next(l))
    [(0, 1), (1, 2)]

    """
    iterable=iter(iterable)
    try:
        cur = next(iterable)
        while True:
            nxt = next(iterable)
            yield (cur, nxt)
            cur = nxt
    except StopIteration:
        pass

def iter_subclasses(cls, _seen=None):
    """
    iter_subclasses(cls)

    Generator over all subclasses of a given class, in depth first order.

    Examples
    --------

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
    if _seen is None:
        _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError:  # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in iter_subclasses(sub, _seen):
                yield sub


class ComparableMixin(object):
    """
    Mixin class that makes instances comparable in a rich way (i.e. in !=, <, <=
    etc), by just implementing a _cmpkey() method that returns a comparable
    value.

    source:
    http://regebro.wordpress.com/2010/12/13/python-implementing-rich-comparison-the-correct-way/

    Examples
    --------

    >>> class MyClass(ComparableMixin):
    ...     def __init__(self, x):
    ...         self.x = x
    ...     def _cmpkey(self):
    ...         return self.x
    >>>
    >>> a = MyClass(3)
    >>> b = MyClass(4)
    >>> a == b
    False
    >>> a < b
    True

    """

    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            # _cmpkey not implemented, or return different type,
            # so I can't compare with "other".
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)


class ReplaceRefMixin(object):
    """
    This class is a utility mixin class to replace references to objects with
    references to other objects. This is useful for example when cloning a
    timeline with a doubly linked list of timepoints, as it updates the `next`
    and `prev` references of the new timepoints with their new neighbors.

    To use this functionality, a class should inherit from this class, and keep
    a list of all attributes that contain references.

    Examples
    --------

    >>> from copy import copy
    >>>
    >>> class MyClass(ReplaceRefMixin):
    ...     def __init__(self, next=None):
    ...         super().__init__()
    ...         self.next = next
    ...         self._ref_attrs.append('next')
    >>>
    >>> a1 = MyClass()
    >>> a2 = MyClass(a1)
    >>> object_map = {}
    >>> b1 = copy(a1)
    >>> b2 = copy(a2)
    >>> object_map[a1] = b1
    >>> object_map[a2] = b2
    >>> b2.next == b1
    False
    >>> b2.next == a1
    True
    >>> b2.replace_refs(object_map)
    >>> b2.next == b1
    True

    """
    def __init__(self):
        self._ref_attrs = []
        
    def replace_refs(self, o_map):
        if hasattr(self, '_ref_attrs'):
            for attr in self._ref_attrs:
                o = getattr(self, attr)
                # if isinstance(o, list):
                #     setattr(self, attr, [o_map.get(o_el) for o_el in o])
                # else:
                #     setattr(self, attr, o_map.get(o))
                if o is None:
                    pass
                elif isinstance(o, list):
                    o_list_new = []

                    for o_el in o:
                        if o_el in o_map:
                            o_list_new.append(o_map[o_el])
                        else:
                            LOGGER.warning(f'reference not found in o_map: {o_el} start={o_el.start} end={o_el.end}, substituting None')
                            # raise
                            o_list_new.append(None)

                    setattr(self, attr, o_list_new)
                else:
                    if o in o_map:
                        o_new = o_map[o]
                    else:
                        print([type(o), o])
                        import partitura.score as score
                        if isinstance(o, score.Note):
                            m =o.start.get_prev_of_type(score.Measure, eq=True)[0]
                            print(m)
                        LOGGER.warning(f'reference not found in o_map: {o} start={o.start} end={o.end}, substituting None')
                        # raise
                        o_new = None
                    setattr(self, attr, o_new)


def partition(func, iterable):
    """
    Return a dictionary containing the equivalence classes (actually bags)
    of iterable, partioned according to func. The value of a key k is the 
    list of all elements e from iterable such that k = func(e)

    Examples
    ========

    >>> l = range(10)
    >>> partition(lambda x: x % 3, l)
    {0: [0, 3, 6, 9], 1: [1, 4, 7], 2: [2, 5, 8]}

    """
    result = defaultdict(list)
    for v in iterable:
        result[func(v)].append(v)
    return dict(result)


def add_field(a, descr):
    """
    Return a new array that is like "a", but has additional fields.

    Source: https://stackoverflow.com/questions/1201817/adding-a-field-to-a-structured-numpy-array

    Parameters
    ----------
    a: np.ndarray
        A structured numpy array
    descr: np.dtype
        A numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    Returns
    -------
    np.ndarray
         The new structured numpy array

    Examples
    --------

    >>> sa = np.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == np.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == np.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> np.all(sa['id'] == sb['id'])
    True
    >>> np.all(sa['name'] == sb['name'])
    True

    """
    if a.dtype.fields is None:
        raise ValueError("`A` must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b


if __name__ == '__main__':
    import doctest
    doctest.testmod()
