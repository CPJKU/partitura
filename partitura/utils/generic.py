#!/usr/bin/env python

import logging
from collections import defaultdict
from textwrap import dedent
import numpy as np

LOGGER = logging.getLogger(__name__)

__all__ = ['find_nearest', 'iter_current_next', 'partition', 'iter_subclasses']


def find_nearest(array, value):
    """
    Return the index of the value in `array` that is closest to `value`.
    
    Parameters
    ----------
    array: ndarray
        Array of numbers
    value: float
        The query value
    
    Returns
    -------
    int
        Index of closest value
    """
    
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) <= np.abs(value - array[idx])):
        return idx-1
    else:
        return idx


# we need a globally unique value to detect whether a keyword argument was
# passed to iter_current_next
_sentinel = object()

def iter_current_next(iterable, start=_sentinel, end=_sentinel):
    """Iterate over pairs of consecutive values in an iterable.

    This creates generator that yields a (current, next) tuple per element. If the
    iterable contains less than two elements a StopIteration exception is
    raised.

    Parameters
    ----------
    iterable: iterable
        Iterable to take values from
    start: object, optional
        If specified, this value will be treated as if it were the first element
        of the iterator
    end: object, optional
        If specified, this value will be treated as if it were the last element
        of the iterator
    
    Yields
    ------
    (object, object)
        Pairs of items

    Examples
    --------

    >>> for pair in iter_current_next([]):
    ...     print(pair)

    >>> for pair in iter_current_next([0]):
    ...     print(pair)

    >>> for pair in iter_current_next([0, 1, 2]):
    ...     print(pair)
    (0, 1)
    (1, 2)

    >>> for pair in iter_current_next([0, 1, 2], start=None):
    ...     print(pair)
    (None, 0)
    (0, 1)
    (1, 2)

    >>> for pair in iter_current_next([0, 1, 2], end='end_value'):
    ...     print(pair)
    (0, 1)
    (1, 2)
    (2, 'end_value')

    >>> for pair in iter_current_next([], start='start', end='end'):
    ...     print(pair)
    ('start', 'end')

    """
    iterable = iter(iterable)
    
    cur = start
    try:

        if cur is _sentinel:
            cur = next(iterable)
            
        while True:

            nxt = next(iterable)
            yield (cur, nxt)
            cur = nxt

    except StopIteration:

        if cur is not _sentinel and end is not _sentinel:
            yield (cur, end)


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


class ReplaceRefMixin(object):
    """This class is a utility mixin class to replace references to
    objects with references to other objects. This is functionality is
    used when unfolding timelines.

    To use this functionality, a class should inherit from this class,
    and keep a list of all attributes that contain references.

    Examples
    --------
    The following class defines `prev` as a referential attribute, to
    be replaced when a class instance is copied:

    >>> class MyClass(ReplaceRefMixin):
    ...     def __init__(self, prev=None):
    ...         super().__init__()
    ...         self.prev = prev
    ...         self._ref_attrs.append('prev')

    Create two instance `a1` and `a2`, where `a1` is the `prev` of
    `a2``

    >>> a1 = MyClass()
    >>> a2 = MyClass(a1)

    Copy `a1` and `a2` to `b1` and `b2`, respectively:

    >>> b1 = copy(a1)
    >>> b2 = copy(a2)

    After copying the prev of `b2` is `a1`, not `b1`:

    >>> b2.prev == b1
    False
    >>> b2.prev == a1
    True

    To fix that we define an object map:

    >>> object_map = {a1: b1, a2: b2}

    and replace the references according to the map:

    >>> b2.replace_refs(object_map)
    >>> b2.prev == b1
    True
    
    """

    def __init__(self):
        self._ref_attrs = []

    def replace_refs(self, o_map):
        if hasattr(self, '_ref_attrs'):
            for attr in self._ref_attrs:
                o = getattr(self, attr)
                if o is None:
                    pass
                elif isinstance(o, list):
                    o_list_new = []

                    for o_el in o:
                        if o_el in o_map:
                            o_list_new.append(o_map[o_el])
                        else:
                            LOGGER.warning(dedent('''reference not found in
                            o_map: {} start={} end={}, substituting None
                            '''.format(o_el, o_el.start, o_el.end)))
                            o_list_new.append(None)

                    setattr(self, attr, o_list_new)
                else:
                    if o in o_map:
                        o_new = o_map[o]
                    else:
                        LOGGER.warning(dedent('''reference not found in o_map:
                        {} start={} end={}, substituting None
                        '''.format(o, o.start, o.end)))
                        o_new = None
                    setattr(self, attr, o_new)


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


def partition(func, iterable):
    """
    Return a dictionary containing the equivalence classes (actually bags)
    of `iterable`, partioned according to `func`. The value of a key `k` is the
    list of all elements `e` from iterable such that `k = func(e)`.

    Examples
    ========

    The following example groups the integers from 0 to 10 by their respective modulo 3 values:

    >>> lst = range(10)
    >>> partition(lambda x: x % 3, lst)
    {0: [0, 3, 6, 9], 1: [1, 4, 7], 2: [2, 5, 8]}

    """
    result = defaultdict(list)
    for v in iterable:
        result[func(v)].append(v)
    return dict(result)


def add_field(a, descr):
    """
    Return a new array that is like `a`, but has additional fields.
    The contents of `a` are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    Source: https://stackoverflow.com/questions/1201817/adding-a-field-to-a-structured-numpy-array

    Parameters
    ----------
    a: np.ndarray
        A structured numpy array
    descr: np.dtype
        A numpy type description of the new fields

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


def sorted_dict_items(items, key=None):
    for item in sorted(items, key=key):
        yield item


def show_diff(a, b):
    """
    Show the difference between two strings, using the difflib package. The
    difference is printed to stdout.

    Parameters
    ----------
    a: str
        First string
    b: str
        Second string
    """

    import difflib
    differ = difflib.Differ()
    for l in differ.compare(a.split('\n'), b.split('\n')):
        print(l)


class PrettyPrintTree(object):
    def __init__(self):
        self.stack = []

    def push(self):
        self.stack.append(TreeSymbol())

    def pop(self):
        self.stack.pop()

    def next_item(self):
        assert len(self.stack) > 0
        self.stack[-1].next_item()

    def last_item(self):
        assert len(self.stack) > 0
        self.stack[-1].last_item()

    def __str__(self):
        return ''.join(str(sym) for sym in self.stack)


class TreeSymbol(object):
    def __init__(self):
        self.symbols = [' │  ', ' ├─ ', ' └─ ', '    ']
        self.state = 0

    def next_item(self):
        self.state = 1

    def last_item(self):
        self.state = 2

    def __str__(self):
        sym = self.symbols[self.state]
        if self.state == 1:
            self.state = 0
        elif self.state == 2:
            self.state = 3
        return sym


def search(states, success, expand, combine):
    while len(states) > 0:
        state = states.pop(0)
        if success(state):
            return state
        else:
            states = combine(expand(state), states)


# def search_recursive(states, success, expand, combine):
#     try:
#         if not states:
#             return None
#         elif success(states[0]):
#             return states[0]
#         else:
#             new_states = combine(expand(states[0]), states[1:])
#             return search_recursive(new_states, success, expand, combine)
#     except RecursionError:
#         warnings.warn('search exhausted stack, bailing out')
#         return None


if __name__ == '__main__':
    import doctest
    doctest.testmod()
