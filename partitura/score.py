#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This module contains an ontology to represent musical scores. A score is
defined at the highest level by a `Part` object (or a hierarchy of `Part` objects,
in a `PartGroup` object). This object contains a `TimeLine` object, which as acts
as a washing line for the elements in a musical score such as measures, notes,
slurs, words, expressive directions. The `TimeLine` object contains a sequence of
`TimePoint` objects, which are the pegs that fix the score elements in time. Each
`TimePoint` object has a time value `t` (an integer). Furthermore, it contains
dictionaries with the objects starting and ending at `t`, respectively (indexed
by class).

TODO: Refer to documentation


"""

import sys
import string
import re
from copy import copy
from textwrap import dedent
from collections import defaultdict
import warnings
import logging
import operator
import itertools
from numbers import Number

import numpy as np
from scipy.interpolate import interp1d

from partitura.utils import (
    ComparableMixin,
    partition,
    iter_subclasses,
    iter_current_next,
    sorted_dict_items,
    PrettyPrintTree,
    MIDI_BASE_CLASS,
    ALTER_SIGNS,
    format_symbolic_duration,
    estimate_symbolic_duration,
    fifths_mode_to_key_name
)
# the score ontology for longer scores requires a high recursion limit
# increase when needed
# sys.setrecursionlimit(10000)

LOGGER = logging.getLogger(__name__)

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
                        if isinstance(o, Note):
                            m = o.start.get_prev_of_type(Measure, eq=True)[0]
                        LOGGER.warning(dedent('''reference not found in o_map:
                        {} start={} end={}, substituting None
                        '''.format(o, o.start, o.end)))
                        o_new = None
                    setattr(self, attr, o_new)


class TimeLine(object):

    """
    The `TimeLine` class collects `TimePoint` objects in a doubly
    linked list fashion (as well as in an array).

    Attributes
    ----------
    points : ndarray of TimePoint objects
        an array of TimePoint objects.

    """

    def __init__(self):
        self.points = np.array([], dtype=TimePoint)
        self._quarter_times = [0]
        self._quarter_durations = [1]
        self._make_quarter_map()

    def quarter_durations(self, start=-np.inf, end=np.inf):
        qd = np.column_stack((self._quarter_times, self._quarter_durations))
        qd = qd[qd[:, 0]>=start, :]
        qd = qd[qd[:, 0]<end, :]
        return qd

    # @property
    # def quarter_durations(self):
    #     return np.column_stack((self._quarter_times, self._quarter_durations))
    
    def _make_quarter_map(self):
        x = self._quarter_times
        y = self._quarter_durations
        if len(x) == 1:
            x = x + x
            y = y + y
        self._quarter_map = interp1d(x, y,
                                     kind='previous',
                                     bounds_error=False,
                                     fill_value=(y[0], y[-1]))

    def set_quarter_duration(self, t, quarter):
        # add quarter duration at time t, unless it is redundant. If another
        # quarter duration is at t, replace it.
        times = self._quarter_times
        quarters = self._quarter_durations
        i = np.searchsorted(self._quarter_times, t)
        changed = False
        i_prev = i - 1
        if (i_prev < 0
            or quarters[i_prev] != quarter):
            # add or replace
            if i == len(times) or times[i] != t:
                # add
                self._quarter_times.insert(i, t)
                self._quarter_durations.insert(i, quarter)
                changed = True
            elif quarters[i] != quarter:
                # replace
                self._quarter_durations[i] = quarter
                changed = True

        if not changed:
            return 

        self._make_quarter_map()
        
        if i + 1 == len(self._quarter_times):
            t_next = np.inf
        else:
            t_next = self._quarter_times[i+1]

        # update quarter attribute of all timepoints in the range [t, t_next]
        start_idx = np.searchsorted(self.points, TimePoint(t))
        end_idx = np.searchsorted(self.points, TimePoint(t_next))
        for tp in self.points[start_idx:end_idx]:
            tp.quarter = quarter

    def _add_point(self, tp):
        """
        add `TimePoint` object `tp` to the time line, unless there is already a timepoint at the same time
        """
        i = np.searchsorted(self.points, tp)
        if i == len(self.points) or self.points[i].t != tp.t:
            self.points = np.insert(self.points, i, tp)
            if i > 0:
                self.points[i - 1].next = self.points[i]
                self.points[i].prev = self.points[i - 1]
            if i < len(self.points) - 1:
                self.points[i].next = self.points[i + 1]
                self.points[i + 1].prev = self.points[i]

    def _remove_point(self, tp):
        """Remove `TimePoint` object `tp` from the time line

        """
        i = np.searchsorted(self.points, tp)
        if self.points[i] == tp:
            self.points = np.delete(self.points, i)
            if i > 0:
                self.points[i - 1].next = self.points[i]
                self.points[i].prev = self.points[i - 1]
            if i < len(self.points) - 1:
                self.points[i].next = self.points[i + 1]
                self.points[i + 1].prev = self.points[i]

    def get_point(self, t):
        """Return the `TimePoint` object with time `t`, or None if there is no
        such object.

        """
        i = np.searchsorted(self.points, TimePoint(t))
        if i < len(self.points) and self.points[i].t == t:
            return self.points[i]
        else:
            return None

    def get_or_add_point(self, t):
        """Return the `TimePoint` object with time `t`; if there is no
        such object, create it, add it to the time line, and return
        it.

        Parameters
        ----------
        t : int
            time value `t`

        Returns
        -------
        TimePoint
            a TimePoint object with time `t`

        """

        tp = self.get_point(t)
        if tp is None:
            tp = TimePoint(t, int(self._quarter_map(t)))
            self._add_point(tp)
        return tp

    def add(self, o, start=None, end=None):
        if start is not None:
            self.get_or_add_point(start).add_starting_object(o)
        if end is not None:
            self.get_or_add_point(end).add_ending_object(o)
            
        
    def remove(self, o, which='both'):
        """Remove `o` from the timeline

        Parameters
        ----------
        o : TimedObject
            Object to be removed
        which : {'start', 'end', 'both'}, optional
            Whether to remove o as a starting object, an ending
            object, or both
        
        """
        if which in ('start', 'both') and o.start:
            try:
                o.start.starting_objects[o.__class__].remove(o)
            except:
                raise Exception('Not implemented: removing an object that is registered by its superclass')
            # cleanup timepoint if no starting/ending objects are left
            if (sum(len(oo) for oo in o.start.starting_objects.values()) +
                sum(len(oo) for oo in o.start.ending_objects.values())) == 0:
                self._remove_point(o.start)
            o.start = None

        if which in ('end', 'both') and o.end:
            try:
                o.end.ending_objects[o.__class__].remove(o)
            except:
                raise Exception('Not implemented: removing an object that is registered by its superclass')
            # cleanup timepoint if no starting/ending objects are left
            if (sum(len(oo) for oo in o.end.starting_objects.values()) +
                sum(len(oo) for oo in o.end.ending_objects.values())) == 0:
                self._remove_point(o.end)
            o.end = None
        
    # def get_all(self, cls, start=None, end=None, include_subclasses=False, mode='starting'):
    #     """Return a list of all instances of `cls` that either start or
    #     end (depending on `mode`) in the interval `start` to `end`.
    #     When `start` and `end` are omitted, the whole timeline is
    #     searched.

    #     Parameters
    #     ----------
    #     cls : class
    #         The class to search for
    #     start : TimePoint, optional
    #         The start of the interval to search. If omitted or None,
    #         the search starts at the start of the timeline. Defaults
    #         to None.
    #     end : TimePoint, optional
    #         The end of the interval to search. If omitted or None, the
    #         search ends at the end of the timeline. Defaults to None.
    #     include_subclasses : bool, optional
    #         If True also return instances that are subclasses of
    #         `cls`. Defaults to False.
    #     mode : {'starting', 'ending'}
    #         Flag indicating whether to search for starting or ending
    #         objects. Defaults to 'starting'.

    #     Returns
    #     -------
    #     list
    #         List of instances of `cls`
        
    #     """
    #     if not mode in ('starting', 'ending'):
    #         LOGGER.warning('unknown mode "{}", using "starting" instead'.format(mode))
    #         mode = 'starting'

    #     if start is not None:
    #         if not isinstance(start, TimePoint):
    #             start = TimePoint(start)

    #         start_idx = np.searchsorted(
    #             self.points, start, side='left')
    #     else:
    #         start_idx = 0

    #     if end is not None:
    #         if not isinstance(end, TimePoint):
    #             end = TimePoint(end)
    #         end_idx = np.searchsorted(self.points, end, side='left')
    #     else:
    #         end_idx = len(self.points)

    #     r = []
    #     if mode == 'ending':
    #         for tp in self.points[start_idx: end_idx]:
    #             r.extend(tp.get_ending_objects_of_type(cls, include_subclasses))
    #     else:
    #         for tp in self.points[start_idx: end_idx]:
    #             r.extend(tp.get_starting_objects_of_type(cls, include_subclasses))
    #     return r

    def iter_all(self, cls, start=None, end=None,
                 include_subclasses=False, mode='starting'):
        """Iterate (in direction of increasing time) over all instances
        of `cls` that either start or end (depending on `mode`) in the
        interval `start` to `end`.  When `start` and `end` are
        omitted, the whole timeline is searched.

        Parameters
        ----------
        cls : class
            The class to search for
        start : TimePoint, optional
            The start of the interval to search. If omitted or None,
            the search starts at the start of the timeline. Defaults
            to None.
        end : TimePoint, optional
            The end of the interval to search. If omitted or None, the
            search ends at the end of the timeline. Defaults to None.
        include_subclasses : bool, optional
            If True also return instances that are subclasses of
            `cls`. Defaults to False.
        mode : {'starting', 'ending'}
            Flag indicating whether to search for starting or ending
            objects. Defaults to 'starting'.

        Returns
        -------
        list
            List of instances of `cls`
        
        """
        if not mode in ('starting', 'ending'):
            LOGGER.warning('unknown mode "{}", using "starting" instead'.format(mode))
            mode = 'starting'

        if start is None:
            start_idx = 0
        else:
            if not isinstance(start, TimePoint):
                start = TimePoint(start)
            start_idx = np.searchsorted(self.points, start)

        if end is None:
            end_idx = len(self.points)
        else:
            if not isinstance(end, TimePoint):
                end = TimePoint(end)
            end_idx = np.searchsorted(self.points, end)


        r = []
        if mode == 'ending':
            for tp in self.points[start_idx: end_idx]:
                yield from tp.iter_ending(cls, include_subclasses)
        else:
            for tp in self.points[start_idx: end_idx]:
                yield from tp.iter_starting(cls, include_subclasses)
        return r


    @property
    def last_point(self):
        """The last TimePoint on the timeline, or None if the timeline is
        empty.
        
        """
        return self.points[-1] if len(self.points) > 0 else None

    @property
    def first_point(self):
        """The first TimePoint on the timeline, or None if the timeline
        is empty.
        
        """
        return self.points[0] if len(self.points) > 0 else None


class TimePoint(ComparableMixin):

    """A TimePoint represents an position on a TimeLine.

    The `TimeLine` class stores sorted TimePoint objects in an array under
    TimeLine.points, as well as doubly linked (through the `prev` and
    `next` attributes). The `TimeLine` class also has functionality to add,
    and remove TimePoints.

    Parameters
    ----------
    t : int
        Time point of some event in/element of the score, where the unit of
        a time point is the <divisions> as defined in the musicxml file,
        more precisely in the corresponding score part. Represents the
        absolute time of the time point, also used for ordering TimePoint
        objects w.r.t. each other.
    label : str, optional
        Default: ''

    Attributes
    ----------
    t : int
    starting_objects : dictionary
        a dictionary where the musical objects starting at this time are
        grouped by class.
    ending_objects : dictionary
        a dictionary where the musical objects ending at this time are
        grouped by class.
    prev
        the preceding time instant (or None if there is none)
    next
        The succeding time instant (or None if there is none)

    """

    def __init__(self, t, quarter=None):
        self.t = t
        self.quarter = quarter
        self.starting_objects = defaultdict(list)
        self.ending_objects = defaultdict(list)
        # prev and next are dynamically updated once the timepoint is part of a timeline
        self.next = None
        self.prev = None

    def __iadd__(self, value):
        assert isinstance(value, Number)
        self.t += value
        return self

    def __isub__(self, value):
        assert isinstance(value, Number)
        self.t -= value
        return self

    def __add__(self, value):
        assert isinstance(value, Number)
        new = copy(self)
        new += value
        return new

    def __sub__(self, value):
        assert isinstance(value, Number)
        new = copy(self)
        new -= value
        return new

    def __str__(self):
        return 'Timepoint {}'.format(self.t)

    def add_starting_object(self, obj):
        """Add object `obj` to the list of starting objects

        """
        obj.start = self
        self.starting_objects[type(obj)].append(obj)

    def add_ending_object(self, obj):
        """Add object `obj` to the list of ending objects

        """
        obj.end = self
        self.ending_objects[type(obj)].append(obj)

    def iter_starting(self, otype, include_subclasses=False):
        """Iterate over all objects of type `otype` that start at this
        time point
        
        """
        yield from self.starting_objects[otype]
        if include_subclasses:
            for subcls in iter_subclasses(otype):
                yield from self.starting_objects[subcls]

    def iter_ending(self, otype, include_subclasses=False):
        """Iterator over all objects of type `otype` that end at this
        time point
        
        """
        yield from self.ending_objects[otype]
        if include_subclasses:
            for subcls in iter_subclasses(otype):
                yield from self.ending_objects[subcls]

    # OBSOLETE
    def get_starting_objects_of_type(self, otype, include_subclasses=False):
        """Return all objects of type `otype` that start at this time point

        """
        # if include_subclasses:
        #     return self.starting_objects[otype] + \
        #         list(itertools.chain(*(self.starting_objects[subcls]
        #                                for subcls in iter_subclasses(otype))))
        # else:
        #     return self.starting_objects[otype]
        yield from self.starting_objects[otype]
        if include_subclasses:
            for subcls in iter_subclasses(otype):
                yield from self.starting_objects[subcls]

    # OBSOLETE
    def get_ending_objects_of_type(self, otype, include_subclasses=False):
        """Return all objects of type `otype` that end at this time point

        """
        # if include_subclasses:
        #     return self.ending_objects[otype] + \
        #         list(itertools.chain(*(self.ending_objects[subcls]
        #                                for subcls in iter_subclasses(otype))))
        # else:
        #     return self.ending_objects[otype]
        yield from self.ending_objects[otype]
        if include_subclasses:
            for subcls in iter_subclasses(otype):
                yield from self.ending_objects[subcls]

    def get_prev_of_type(self, otype, eq=False):
        """Return the object(s) of type `otype` that start at the latest time
        before this time point (or at this time point, if `eq` is True)

        """
        if eq:
            tp = self
        else:
            tp = self.prev

        while tp:
            yield from tp.get_starting_objects_of_type(otype)
            tp = tp.prev

    def get_next_of_type(self, otype, eq=False):
        """Return the object(s) of type `otype` that start at the earliest
        time after this time point (or at this time point, if `eq` is True)

        """
        if eq:
            tp = self
        else:
            tp = self.next

        while tp:
            yield from tp.get_starting_objects_of_type(otype)
            tp = tp.next


    def _cmpkey(self):
        """This method returns the value to be compared (code for that is in
        the ComparableMixin class)

        """
        return self.t

    def _pp(self, tree):
        result = ['{}Timepoint {}'.format(tree, self.t)]
        tree.push()

        ending_items_lists = sorted_dict_items(self.ending_objects.items(),
                                               key=lambda x: x[0].__name__)
        starting_items_lists = sorted_dict_items(self.starting_objects.items(),
                                                 key=lambda x: x[0].__name__)

        ending_items = [o for _, oo in ending_items_lists for o in oo]
        starting_items = [o for _, oo in starting_items_lists for o in oo]

        if ending_items:

            result.append('{}'.format(tree))

            if starting_items:
                tree.next_item()
            else:
                tree.last_item()

            result.append('{}ending objects'.format(tree))
            tree.push()
            result.append('{}'.format(tree))

            for i, item in enumerate(ending_items):

                if i == (len(ending_items) - 1):
                    tree.last_item()
                else:
                    tree.next_item()

                result.append('{}{}'.format(tree, item))

            tree.pop()

        if starting_items:

            result.append('{}'.format(tree))
            tree.last_item()
            result.append('{}starting objects'.format(tree))
            tree.push()
            result.append('{}'.format(tree))

            for i, item in enumerate(starting_items):

                if i == (len(starting_items) - 1):
                    tree.last_item()
                else:
                    tree.next_item()

                result.append('{}{}'.format(tree, item))

            tree.pop()

        tree.pop()
        return result


class TimedObject(ReplaceRefMixin):

    """Class that represents objects that (may?) have a start and ending
    point. Used as super-class for classes representing different types of
    objects in a (printed) score.

    """

    def __init__(self):
        super().__init__()
        self.start = None
        self.end = None


class Page(TimedObject):

    def __init__(self, nr=0):
        super().__init__()
        self.nr = nr

    def __str__(self):
        return 'Page: number={0}'.format(self.nr)


class System(TimedObject):

    def __init__(self, nr=0):
        super().__init__()
        self.nr = nr

    def __str__(self):
        return 'System: number={0}'.format(self.nr)

class Clef(TimedObject):
    """
    TODO: explain the connection between the number of the clef and the number
    in the staff attribute of Notes etc.
    """
    def __init__(self, number, sign, line, octave_change):
        super().__init__()
        self.number = number
        self.sign = sign
        self.line = line
        self.octave_change = octave_change

    def __str__(self):
        return 'Clef: sign={} line={} number={}'.format(self.sign, self.line, self.number)


class Slur(TimedObject):

    def __init__(self, start_note=None, end_note=None):
        super().__init__()
        self.start_note = start_note
        self.end_note = end_note
        # maintain a list of attributes to update when cloning this instance
        self._ref_attrs.extend(['start_note', 'end_note'])

    def __str__(self):
        # return 'slur at voice {0} (ends at {1})'.format(self.voice, self.end and self.end.t)
        start = '' if self.start_note is None else 'start={}'.format(self.start_note.id)
        end = '' if self.end_note is None else 'end={}'.format(self.end_note.id)
        return ' '.join(('Slur', start, end)).strip()


class Repeat(TimedObject):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Repeat (from {0} to {1})'.format(self.start and self.start.t, self.end and self.end.t)


class DaCapo(TimedObject):

    def __str__(self):
        return u'Dacapo'


class Fine(TimedObject):

    def __str__(self):
        return 'Fine'


class Fermata(TimedObject):

    def __init__(self, ref=None):
        super().__init__()
        # ref(erent) can be a note or a barline
        self.ref = ref

    def __str__(self):
        return 'Fermata ref={}'.format(self.ref)


class Ending(TimedObject):

    """Class that represents one part of a 1---2--- type ending of a musical
    passage (a.k.a Volta brackets).

    """

    def __init__(self, number):
        super().__init__()
        self.number = number

    def __str__(self):
        return 'Ending (from {0} to {1})'.format(self.start.t, self.end.t)


class Measure(TimedObject):

    """Measure.

    Attributes
    ----------
    number : int
        the number of the measure. (directly taken from musicxml file?)
    page : int
    system : int
    incomplete : boolean

    """

    def __init__(self, number=None):
        super().__init__()
        self.number = number


    def __str__(self):
        # return 'Measure {0} at page {1}, system {2}'.format(self.number, self.page, self.system)
        return 'Measure: number={0}'.format(self.number)

    @property
    def page(self):
        pages = self.start.get_prev_of_type(Page, eq=True)
        if pages:
            return pages[0].nr
        else:
            return None

    @property
    def system(self):
        systems = self.start.get_prev_of_type(System, eq=True)
        if systems:
            return systems[0].nr
        else:
            return None

    # def get_measure_duration(self, quarter=False):
    #     """Return the measure duration, either in beats or in quarter note
    #     units.

    #     Parameters
    #     ----------
    #     quarter : bool (optional)
    #         If True, return the measure duration in quarter note units,
    #         otherwise in beat units. Defaults to False

    #     Returns
    #     -------
    #     float
    #         The measure duration

    #     """

    #     # TODO: support mid-measure time sig and division changes
    #     divs = self.start.get_prev_of_type(Divisions, eq=True)
    #     ts = self.start.get_prev_of_type(TimeSignature, eq=True)
    #     assert len(divs) > 0
    #     assert len(ts) > 0
    #     measure_dur = self.end.t - self.start.t
    #     beats = ts[0].beats
    #     beat_type = ts[0].beat_type
    #     div = float(divs[0].divs)

    #     if quarter:
    #         return measure_dur / div
    #     else:
    #         return beat_type * measure_dur / (4. * div)

    # @property
    # def incomplete(self):
    #     """Returns True if the duration of the measure is less than the
    #     expected duration (as computed based on the current divisions and
    #     time signature), and False otherwise.

    #     Returns
    #     -------
    #     bool

    #     """

    #     # divs = self.start.get_prev_of_type(Divisions, eq=True)
    #     ts = self.start.get_prev_of_type(TimeSignature, eq=True)

    #     invalid = False
    #     if len(divs) == 0:
    #         LOGGER.warning('Part specifies no divisions')
    #         invalid = True
    #     if len(ts) == 0:
    #         LOGGER.warning('Part specifies no time signatures')
    #         invalid = True

    #     if invalid:
    #         LOGGER.warning(
    #             'could not be determine if meaure is incomplete, assuming complete')
    #         return False

    #     # measure_dur = nextm[0].start.t - self.start.t
    #     measure_dur = self.end.t - self.start.t
    #     beats = ts[0].beats
    #     beat_type = ts[0].beat_type
    #     div = float(divs[0].divs)

    #     # this will return a boolean, so either True or False
    #     # return beat_type * measure_dur / (4 * div * beats) % 1 > 0
    #     return beat_type*measure_dur < 4*div*beats


class TimeSignature(TimedObject):

    """
    Parameters
    ----------
    beats :
    beat_type :
    """

    def __init__(self, beats, beat_type):
        super().__init__()
        self.beats = beats
        self.beat_type = beat_type

    def __str__(self):
        return 'Time signature: {0}/{1}'.format(self.beats, self.beat_type)


class Tempo(TimedObject):

    def __init__(self, bpm, unit=None):
        super().__init__()
        self.bpm = bpm
        self.unit = unit

    def __str__(self):
        if self.unit:
            return 'Tempo: {}={}'.format(self.unit, self.bpm)
        else:
            return 'Tempo: bpm={0}'.format(self.bpm)


class KeySignature(TimedObject):
    """Key signature.

    Parameters
    ----------
    fifths : number
        Number of sharps (positive) or flats (negative)
    mode : str
        Mode of the key, either 'major' or 'minor'

    Attributes
    ----------
    name

    """

    def __init__(self, fifths, mode):
        super().__init__()
        self.fifths = fifths
        self.mode = mode

    @property
    def name(self):
        """The key signature name, where the root is uppercase, and an
        trailing 'm' indicates minor modes (e.g. 'Am', 'G#').

        Returns
        -------
        str
            The key signature name

        """
        return fifths_mode_to_key_name(self.fifths, self.mode)

    def __str__(self):
        return ('Key signature: fifths={}, mode={} ({})'
                .format(self.fifths, self.mode, self.name))


class Transposition(TimedObject):

    """Represents a <transpose> tag that tells how to change all (following)
    pitches of that part to put it to concert pitch (i.e. sounding pitch).

    Parameters
    ----------
    diatonic : int
    chromatic : int
        The number of semi-tone steps to add or subtract to the pitch to
        get to the (sounding) concert pitch.

    """

    def __init__(self, diatonic, chromatic):
        super().__init__()
        self.diatonic = diatonic
        self.chromatic = chromatic

    def __str__(self):
        return 'Transposition: diatonic={0}, chromatic={1}'.format(self.diatonic, self.chromatic)


class Words(TimedObject):

    """
    Parameters
    ----------
    text : str
    """

    def __init__(self, text, staff=None):
        super().__init__()
        self.text = text
        self.staff = staff

    def __str__(self):
        return '{}: {}'.format(type(self).__name__, self.text)


class Direction(TimedObject):

    """

    """

    def __init__(self, text, raw_text=None, staff=None):
        super().__init__()
        self.text = text
        self.raw_text = raw_text
        self.staff = staff

    def __str__(self):
        if self.raw_text is not None:
            return '{}: {} ({})'.format(type(self).__name__, self.text, self.raw_text)
        else:
            return '{}: {}'.format(type(self).__name__, self.text)


class TempoDirection(Direction): pass


class DynamicTempoDirection(TempoDirection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.intermediate = []


class ConstantTempoDirection(TempoDirection): pass

class ConstantArticulationDirection(TempoDirection): pass

class ResetTempoDirection(ConstantTempoDirection): pass


class LoudnessDirection(Direction): pass


class DynamicLoudnessDirection(LoudnessDirection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.intermediate = []

class ConstantLoudnessDirection(LoudnessDirection): pass


class ImpulsiveLoudnessDirection(LoudnessDirection): pass


class GenericNote(TimedObject):
    """Represents the common aspects of notes and rests (and in the future
    unpitched notes)

    Parameters
    ----------
    voice : integer, optional (default: None)
    id : integer, optional (default: None)

    """
    def __init__(self, id=None, voice=None, staff=None, symbolic_duration=None, articulations={}):
        self._sym_dur = None
        super().__init__()
        self.voice = voice
        self.id = id
        self.staff = staff
        self.symbolic_duration = symbolic_duration
        self.articulations = articulations

        # these attributes are set after the instance is constructed
        self.fermata = None
        self.tie_prev = None
        self.tie_next = None
        self.slur_stops = []
        self.slur_starts = []

        # maintain a list of attributes to update when cloning this instance
        self._ref_attrs.extend(['tie_prev', 'tie_next', 'slur_stops', 'slur_starts'])

    @property
    def symbolic_duration(self):
        if self._sym_dur is None:
            # compute value
            assert self.start is not None
            assert self.end is not None
            assert self.start.quarter is not None
            return estimate_symbolic_duration(self.duration, self.start.quarter)
        else:
            # return set value
            return self._sym_dur

    @symbolic_duration.setter
    def symbolic_duration(self, v):
        self._sym_dur = v

    @property
    def duration(self):
        """The duration of the note in divisions

        Returns
        -------
        int

        """

        try:
            return self.end.t - self.start.t
        except:
            LOGGER.warn('no end time found for note')
            return 0

    @property
    def end_tied(self):
        """The `Timepoint` corresponding to the end of the note, or---when
        this note belongs to a group of tied notes---the end of the last
        note in the group.

        Returns
        -------
        TimePoint
            End of note

        """
        if self.tie_next is None:
            return self.end
        else:
            return self.tie_next.end_tied

    @property
    def duration_tied(self):
        """Time difference of the start of the note to the end of the note,
        or---when  this note belongs to a group of tied notes---the end of
        the last note in the group.

        Returns
        -------
        int
            Duration of note

        """
        if self.tie_next is None:
            return self.duration
        else:
            return self.duration + self.tie_next.duration_tied

    @property
    def duration_from_symbolic(self):
        if self.symbolic_duration:
            # divs = self.start.get_prev_of_type(Divisions, eq=True)
            # if len(divs) == 0:
            #     div = 1
            # else:
            #     div = divs[0].divs
            return symbolic_to_numeric_duration(self.symbolic_duration, self.start.quarter)
        else:
            return None

    @property
    def tie_prev_notes(self):
        if self.tie_prev:
            return self.tie_prev.tie_prev_notes + [self.tie_prev]
        else:
            return []

    @property
    def tie_next_notes(self):
        if self.tie_next:
            return [self.tie_next] + self.tie_next.tie_next_notes
        else:
            return []

    def __str__(self):
        s = ('{}: id={} voice={} staff={} type={}'
             .format(type(self).__name__, self.id, self.voice, self.staff,
                     format_symbolic_duration(self.symbolic_duration)))
        if len(self.articulations) > 0:
            s += ' articulations=({})'.format(", ".join(self.articulations))
        if self.tie_prev or self.tie_next:
            all_tied = self.tie_prev_notes + [self] + self.tie_next_notes
            tied_dur = '+'.join(format_symbolic_duration(n.symbolic_duration) for n in all_tied)
            tied_id = '+'.join(n.id or 'None' for n in all_tied)
            return s + ' tied: {}'.format(tied_id)
        else:
            return s


class Note(GenericNote):
    def __init__(self, step, alter, octave, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = step
        self.alter = alter
        self.octave = octave

    def __str__(self):
        return ' '.join((super().__str__(),
                         'pitch={}{}{}'.format(self.step, self.alter_sign, self.octave)))

    @property
    def midi_pitch(self):
        """The midi pitch value of the note (MIDI note number). C4 (middle C,
        in german: c') is note number 60.

        Returns
        -------
        integer
            The note's pitch as MIDI note number.

        """
        return ((self.octave + 1) * 12
                + MIDI_BASE_CLASS[self.step.lower()]
                + (self.alter or 0))

    # this will be replaced by carlos
    # @property
    # def morphetic_pitch(self):
    #     """The morphetic value of the note, i.e. a single integer. It
    #     corresponds to the (vertical) position of the note in the barline
    #     system.

    #     Returns
    #     -------
    #     integer

    #     """
    #     return (_MORPHETIC_OCTAVE[self.octave] +
    #             _MORPHETIC_BASE_CLASS[self.step.lower()])

    @property
    def alter_sign(self):
        """The alteration of the note

        Returns
        -------
        str

        """
        return ALTER_SIGNS[self.alter]


    @property
    def previous_notes_in_voice(self):
        n = self
        while True:
            nn = n.start.get_prev_of_type(GenericNote)
            if nn:
                voice_notes = [m for m in nn if m.voice == self.voice]
                if len(voice_notes) > 0:
                    return voice_notes
                n = nn[0]
            else:
                return []

    @property
    def simultaneous_notes_in_voice(self):
        return [m for m in self.start.starting_objects[GenericNote]
                if m.voice == self.voice and m != self]

    @property
    def next_notes_in_voice(self):
        n = self
        while True:
            nn = n.start.get_next_of_type(GenericNote)
            if nn:
                voice_notes = [m for m in nn if m.voice == self.voice]
                if len(voice_notes) > 0:
                    return voice_notes
                n = nn[0]
            else:
                return []


class Rest(GenericNote):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GraceNote(Note):
    def __init__(self, grace_type, *args, steal_proportion=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grace_type = grace_type
        self.steal_proportion = steal_proportion


class PartGroup(object):
    """Represents a grouping of several instruments, usually named, and
    expressed in the score with a group symbol such as a brace or a
    bracket. In symphonic scores, bracketed part groups usually group
    families of instruments, such as woodwinds or brass, whereas
    braces are often used to group multiple instances of the same
    instrument. See the `MusicXML documentation
    <https://usermanuals.musicxml.com/MusicXML/Content/ST-MusicXML-
    group-symbol-value.htm>`_ for further information.

    Parameters
    ----------
    group_symbol : str or None, optional
        The symbol used for grouping instruments.

    Attributes
    ----------
    group_symbol : str or None

    children : list of Part or PartGroup objects

    parent : PartGroup

    number : int
    
    """

    def __init__(self, group_symbol=None, name=None):
        self.group_symbol = group_symbol
        self.children = []
        self.name = name
        self.parent = None
        self.number = None

    def _pp(self, tree):
        result = ['{}PartGroup: name="{}" group_symbol="{}"'
                  .format(tree, self.name, self.group_symbol)]
        tree.push()
        N = len(self.children)
        for i, child in enumerate(self.children):
            result.append('{}'.format(tree))
            if i == N - 1:
                tree.last_item()
            else:
                tree.next_item()
            result.extend(child._pp(tree))
        tree.pop()
        return '\n'.join(result)

    def pretty(self):
        """Return a pretty representation of this object.

        Returns
        -------
        str
            A pretty representation
        
        """
        return self._pp(PrettyPrintTree())

    # def pretty(self, l=0):
    #     if self.name is not None:
    #         name_str = ' / {0}'.format(self.name)
    #     else:
    #         name_str = ''
    #     s = ['    ' * l + '{0}{1}'.format(self.grouping_symbol, name_str)]
    #     for ch in self.children:
    #         s.append(ch.pretty(l + 1))
    #     return '\n'.join(s)


class ScoreVariant(object):
    """
    """

    def __init__(self, timeline, start_time=0):
        self.t_unfold = start_time
        self.segments = []
        self.timeline = timeline

    def add_segment(self, start, end):
        self.segments.append((start, end, self.t_unfold))
        self.t_unfold += (end.t - start.t)

    @property
    def segment_times(self):
        """
        Return segment (start, end, offset) information for each of the segments in
        the score variant.
        """
        return [(s.t, e.t, o) for (s, e, o) in self.segments]

    def __str__(self):
        return 'Segment: {}'.format(self.segment_times)

    def clone(self):
        """
        Return a clone of the ScoreVariant
        """
        clone = ScoreVariant(self.timeline, self.t_unfold)
        clone.segments = self.segments[:]
        return clone

    def create_variant_timeline(self):
        timeline = TimeLine()


        for start, end, offset in self.segments:
            delta = offset - start.t
            qd = self.timeline.quarter_durations(start.t, end.t)
            for t, quarter in qd:
                timeline.set_quarter_duration(t+delta, quarter)
            # After creating the new timeline we need to replace references to
            # objects in the old timeline to references in the new timeline
            # (e.g. t.next, t.prev, note.tie_next). For this we keep track of
            # correspondences between objects (timepoints, notes, measures,
            # etc), in o_map
            o_map = {}
            o_new = set()
            tp = start
            while tp != end:
                # make a new timepoint, corresponding to tp
                tp_new = timeline.get_or_add_point(tp.t+delta)
                o_gen = (o for oo in tp.starting_objects.values() for o in oo)
                for o in o_gen:

                    # special cases:

                    # don't include repeats/endings in the unfolded timeline
                    if isinstance(o, (Repeat, Ending)):
                        continue
                    # # don't repeat divisions if it hasn't changed
                    # elif isinstance(o, Divisions):
                    #     prev = next(iter(tp_new.get_prev_of_type(Divisions)), None)
                    #     if prev is not None and o.divs == prev.divs:
                    #         continue
                    # don't repeat time sig if it hasn't changed
                    elif isinstance(o, TimeSignature):
                        prev = next(iter(tp_new.get_prev_of_type(TimeSignature)), None)
                        if prev is not None and ((o.beats, o.beat_type) ==
                                                 (prev.beats, prev.beat_type)):
                            continue
                    # don't repeat key sig if it hasn't changed
                    elif isinstance(o, KeySignature):
                        prev = next(iter(tp_new.get_prev_of_type(KeySignature)), None)
                        if prev is not None and ((o.fifths, o.mode) ==
                                                 (prev.fifths, prev.mode)):
                            continue

                    # make a copy of the object
                    o_copy = copy(o)
                    # add it to the set of new objects (for which the refs will be replaced)
                    o_new.add(o_copy)
                    # keep track of the correspondence between o and o_copy
                    o_map[o] = o_copy
                    # add the start of the new object to the timeline
                    tp_new.add_starting_object(o_copy)
                    if o.end is not None:
                        # add the end of the object to the timeline
                        tp_end = timeline.get_or_add_point(o.end.t+delta)
                        tp_end.add_ending_object(o_copy)

                tp = tp.next
                if tp is None:
                    raise Exception('segment end not a successor of segment start, invalid score variant')

            # special case: fermata starting at end of segment should be
            # included if it does not belong to a note, and comes at the end of
            # a measure (o.ref == 'right')
            for o in end.starting_objects[Fermata]:
                if o.ref in (None, 'right'):
                    o_copy = copy(o)
                    tp_new = timeline.get_or_add_point(end.t+delta)
                    tp_new.add_starting_object(o_copy)

            # for each of the new objects, replace the references to the old
            # objects to their corresponding new objects
            for o in o_new:
                o.replace_refs(o_map)

        # replace prev/next references in timepoints
        for tp, tp_next in iter_current_next(timeline.points):
            tp.next = tp_next
            tp_next.prev = tp

        return timeline


class Part(object):

    """Represents a score part, e.g. all notes of one single instrument
    (or multiple instruments written in the same staff). Note that
    there may be more than one staff per score part.

    Parameters
    ----------
    id : str
        The identifier of the part. To be compatible with MusicXML the
        identifier should not start with a number
    timeline : TimeLine or None, optional
        If not None, use the provided timeline as timeline for the
        Part. Otherwise a new empty Timeline will be created.

    Attributes
    ----------
    id : str
        The identifier of the part. (see Parameters Section).
    part_name : str
        Name for the part
    part_abbreviation : str
        Abbreviated name for part
    timeline : TimeLine
    notes
    notes_tied
    beat_map
    quarter_map
    
    """

    def __init__(self, id, timeline=None, part_name=None):
        self.id = id
        self.timeline = timeline or TimeLine()
        self.parent = None
        self.part_name = part_name
        self.part_abbreviation = None

    @property
    def part_names(self):
        # get instrument name parts recursively
        chunks = []

        if self.part_name is not None:
            chunks.append(self.part_name)
            yield self.part_name

        part = self.parent
        while part is not None:
            if part.name is not None:
                chunks.insert(0, part.name)
                yield '  '.join(chunks)
            part = part.parent

    def remove(self, obj):
        """
        Remove an object from the timeline.

        Parameters
        ----------
        obj: TimedObject
            Object to be added to the timeline

        """

        self.timeline.remove_starting_object(obj)

        if obj.end is not None:

            self.timeline.remove_ending_object(obj)



    # def test_timeline(self):
    #     """Test if all ending objects have occurred as starting object as
    #     well.

    #     """
    #     return self.timeline.test()


    def _pp(self, tree):
        result = ['{}Part: name="{}" id="{}"'
                  .format(tree, self.part_name, self.id)]
        tree.push()
        N = len(self.timeline.points)
        for i, timepoint in enumerate(self.timeline.points):
            result.append('{}'.format(tree))
            if i == N - 1:
                tree.last_item()
            else:
                tree.next_item()
            result.extend(timepoint._pp(tree))
        tree.pop()
        return '\n'.join(result)

    def pretty(self):
        """Return a pretty representation of this object.

        Returns
        -------
        str
            A pretty representation
        
        """
        return self._pp(PrettyPrintTree())

    @property
    def divisions_map(self):
        qd = self.timeline.quarter_durations()

        start = self.timeline.first_point
        if len(qd) == 0 or (start and qd[0, 0] > start.t):
            qd = np.vstack(([start.t, 1], qd))

        end = self.timeline.last_point
        if end and qd[-1, 0] < end.t:
            qd = np.vstack(([end.t, qd[-1, 1]], qd))

        if len(qd) < 2:
            qd = np.vstack((qd, qd))
            qd[1, 0] += 1

        # return interp1d(qd[:, 0], qd[:, 1], kind='previous',
        #                 bounds_error=False, fill_value=(qd[0, 1], qd[-1, 1]))
        return interp1d(qd[:, 0], qd[:, 1], kind='previous')
        
        # divs = np.array([(divs.start.t, divs.divs) for divs in self.timeline.iter_all(Divisions)])
        # if len(divs) == 0:
        #     # warn assumption
        #     divs = np.array([(self.timeline.first_point.t, 1),
        #                      (self.timeline.last_point.t, 1)])
        # elif divs[0, 0] > self.timeline.first_point.t:
        #     divs = np.vstack(((self.timeline.first_point.t, divs[0, 1]),
        #                       divs))
        # divs = np.vstack((divs,
        #                   (self.timeline.last_point.t, divs[-1, 1])))
        # return interp1d(divs[:, 0], divs[:, 1], kind='previous',
        #                 bounds_error=False, fill_value=(divs[0, 1], divs[-1, 1]))


    @property
    def time_signature_map(self):
        tss = np.array([(ts.start.t, ts.beats, ts.beat_type)
                        for ts in self.timeline.iter_all(TimeSignature)])
        if len(tss) == 0:
            # warn assumption
            tss = np.array([(self.timeline.first_point.t, 4, 4),
                            (self.timeline.last_point.t, 4, 4)])
        elif tss[0, 0] > self.timeline.first_point.t:
            tss = np.vstack(((self.timeline.first_point.t, tss[0, 1], tss[0, 2]),
                             tss))
        tss = np.vstack((tss,
                         (self.timeline.last_point.t, tss[-1, 1], tss[-1, 2])))
        return interp1d(tss[:, 0], tss[:, 1:], kind='previous')


    def _time_interpolator(self, quarter=False, inv=False):
        offset = 0
        keypoints = defaultdict(lambda: [None, None])
        tl = self.timeline
        _ = keypoints[tl.first_point.t]
        _ = keypoints[tl.last_point.t]
        for t, q in zip(tl._quarter_times, tl._quarter_durations):
            keypoints[t][0] = q
        if not quarter:
            for ts in tl.iter_all(TimeSignature):
                # keypoints[ts.start.t][1] = int(np.log2(ts.beat_type))
                keypoints[ts.start.t][1] = ts.beat_type/4
        cur_div = 1
        cur_bt = 1
        keypoints_list = []

        for t in sorted(keypoints.keys()):
            kp = keypoints[t]
            if kp[0] is None:
                kp[0] = cur_div
            else:
                cur_div = kp[0]
            if kp[1] is None:
                kp[1] = cur_bt
            else:
                cur_bt = kp[1]
            if not keypoints_list or kp != keypoints_list[-1]:
                keypoints_list.append([t]+kp)
        keypoints = np.array(keypoints_list, dtype=np.float)

        x = keypoints[:, 0]
        y = np.r_[0, np.cumsum((keypoints[:-1, 2]
                                * np.diff(keypoints[:, 0]))
                               / keypoints[:-1, 1])]
        
        m1 = next(tl.first_point.iter_starting(Measure), None)

        if (m1 and m1.start is not None and m1.end is not None):

            f = interp1d(x, y)
            actual_dur = np.diff(f((m1.start.t, m1.end.t)))[0]
            ts = next(m1.start.iter_starting(TimeSignature), None)

            if ts:

                normal_dur = ts.beats
                if quarter:
                    normal_dur *= ts.beat_type/4
                if actual_dur < normal_dur:
                    y -= actual_dur
            else:
                # warn
                pass

        if len(tl.points) < 2:
            return lambda x: np.zeros(len(x))
        else:
            if inv:
                return interp1d(y, x)
            else:
                return interp1d(x, y)

    @property
    def beat_map(self):
        """A function mapping timeline times to beat times. The function
        can take scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function
        
        """
        return self._time_interpolator()

    @property
    def inv_beat_map(self):
        """A function mapping beat times to timeline times. The function
        can take scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function
        
        """
        return self._time_interpolator(inv=True)


    @property
    def quarter_map(self):
        """A function mapping timeline times to quarter times. The
        function can take scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function
        
        """
        return self._time_interpolator(quarter=True)

    @property
    def inv_quarter_map(self):
        """A function mapping quarter times to timeline times. The
        function can take scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function
        
        """
        return self._time_interpolator(quarter=True, inv=True)

    @property
    def notes(self):
        """Return a list of all Note objects in the part. This list includes
        GraceNote objects but not Rest objects.

        Returns
        -------
        list
            list of Note objects

        """
        return list(self.timeline.iter_all(Note, include_subclasses=True))

    @property
    def notes_tied(self):
        """Return a list of all Note objects in the part that are either not
        tied, or the first note of a group of tied notes. This list
        includes GraceNote objects but not Rest objects.

        Returns
        -------
        list
            list of Note objects

        """
        return [note for note in self.timeline.iter_all(Note, include_subclasses=True)
                if note.tie_prev is None]



def iter_unfolded_timelines(timeline):
    """
    Description
    
    Parameters
    ----------
    timeline: type
        Description of `timeline`
    
    Returns
    -------
    type
        Description of return value
    """
    
    for sv in make_score_variants(timeline):
        yield sv.create_variant_timeline()


def unfold_timeline_maximal(timeline):
    """Return the "maximally" unfolded timeline, that is, a copy of the
    timeline where all segments marked with repeat signs are included
    twice.

    Returns
    -------
    TimeLine
        The unfolded TimeLine

    """

    sv = make_score_variants(timeline)[-1]
    return sv.create_variant_timeline()


def make_score_variants(timeline):
    """Create a list of ScoreVariant objects, each representing a
    distinct way to unfold the score, based on the repeat structure.

    Parameters
    ----------
    timeline: TimeLine
        A timeline for which to make the score variants

    Returns
    -------
    list
        List of ScoreVariant objects

    Notes
    -----
    This function does not currently support nested repeats, such as in
    case 45d of the MusicXML Test Suite.

    """

    if len(list(timeline.iter_all(DaCapo)) +
           list(timeline.iter_all(Fine))) > 0:
        LOGGER.warning(('Generation of repeat structures involving da '
                        'capo/fine/coda/segno directions is not '
                        'supported yet'))

    # TODO: check if we need to wrap in list
    repeats = list(timeline.iter_all(Repeat))
    # repeats may not have start or end times. `_repeats_to_start_end`
    # returns the start/end paisr for each repeat, making educated guesses
    # when these are missing.
    repeat_start_ends = _repeats_to_start_end(repeats,
                                             timeline.first_point,
                                             timeline.last_point)

    # check for nestings and raise if necessary
    if any(n < c for c, n in iter_current_next(repeat_start_ends)):
        raise NotImplementedError('Nested endings are currently not supported')

    # t_score is used to keep the time in the score
    t_score = timeline.first_point
    svs = [ScoreVariant(timeline)]
    # each repeat holds start and end time of a score interval to
    # be repeated
    for i, (rep_start, rep_end) in enumerate(repeat_start_ends):
        new_svs = []
        for sv in svs:
            # is the start of the repeat after our current score
            # position?
            if rep_start > t_score:
                # yes: add the tuple (t_score, rep_start) to the
                # result this is the span before the interval that is
                # to be repeated
                sv.add_segment(t_score, rep_start)

            # create a new ScoreVariant for the repetition (sv will be the
            # score variant where this repeat is played only once)
            new_sv = sv.clone()

            # get any "endings" (e.g. 1 / 2 volta) of the repeat
            # (there are not supposed to be more than one)
            ending1 = next(rep_end.iter_ending(Ending), None)
            # is there an ending?
            if ending1:

                # add the first occurrence of the repeat
                sv.add_segment(rep_start, ending1.start)

                ending2 = next(rep_end.iter_starting(Ending), None)
                if ending2:
                    # add the first occurrence of the repeat
                    sv.add_segment(ending2.start, ending2.end)
                else:
                    # ending 1 without ending 2, should not happen normally
                    pass

                # new_sv includes the 1/2 ending repeat, which means:
                # 1. from repeat start to repeat end (which includes ending 1)
                new_sv.add_segment(rep_start, rep_end)
                # 2. from repeat start to ending 1 start
                new_sv.add_segment(rep_start, ending1.start)
                # 3. ending 2 start to ending 2 end
                new_sv.add_segment(ending2.start, ending2.end)

                # update the score time
                t_score = ending2.end

            else:
                # add the first occurrence of the repeat
                sv.add_segment(rep_start, rep_end)

                # no: add the full interval of the repeat (the second time)
                new_sv.add_segment(rep_start, rep_end)
                new_sv.add_segment(rep_start, rep_end)

                # update the score time
                t_score = rep_end

            # add both score variants
            new_svs.append(sv)
            new_svs.append(new_sv)

        svs = new_svs

    # are we at the end of the piece already?
    if t_score < timeline.last_point:
        # no, append the interval from the current score
        # position to the end of the piece
        for sv in svs:
            sv.add_segment(t_score, timeline.last_point)

    return svs


def add_measures(part):
    """Add measures to a part.

    This function adds Measure objects to the part according to any
    time signatures present in the part. Please note that any existing
    measures will be untouched and ignored.

    The part object will be modified in place.

    Parameters
    ----------
    part : Part
        Part instance
    
    """
    tl = part.timeline
    timesigs = np.array([(ts.start.t, ts.beats)
                         for ts in tl.iter_all(TimeSignature)],
                        dtype=np.int)
    start = tl.first_point.t
    end = tl.last_point.t

    
    # make sure we cover time from the start of the timeline
    if len(timesigs) == 0 or timesigs[0, 0] > start:
        timesigs = np.vstack([[start, 4]], timesigs)

    # in unlikely case of timesig at last point, remove it
    if timesigs[-1, 0] >= end:
        timesigs = timesigs[:-1]

    ts_start_times = timesigs[:, 0]
    beats_per_measure = timesigs[:, 1]
    ts_end_times = ts_start_times[1:]

    # make sure we cover time until the end of the timeline
    if len(ts_end_times) == 0 or ts_end_times[-1] < end:
        ts_end_times = np.r_[ts_end_times, end]

    assert len(ts_start_times) == len(ts_end_times)
    
    beat_map = part.beat_map
    inv_beat_map = part.inv_beat_map
    mcounter = 1

    for ts_start, ts_end, measure_dur in zip(ts_start_times, ts_end_times, beats_per_measure):
        pos = ts_start
    
        while pos < ts_end:
    
            measure_start = pos
            measure_end_beats = min(beat_map(pos)+measure_dur, beat_map(end))
            measure_end = min(ts_end, inv_beat_map(measure_end_beats))
            tl.add(Measure(number=mcounter), int(measure_start), int(measure_end))
            pos = measure_end
            mcounter += 1


def remove_grace_notes(timeline):
    """Remove all grace notes from a timeline.

    The specified timeline object will be modified in place.

    Parameters
    ----------
    timeline : Timeline
        The timeline from which to remove the grace notes
    
    """
    for point in timeline.points:
        point.starting_objects[Note] = [n for n in point.starting_objects[Note]
                                        if n.grace_type is None]
        point.ending_objects[Note] = [n for n in point.ending_objects[Note]
                                      if n.grace_type is None]


def expand_grace_notes(timeline, default_type='appoggiatura', min_steal=.05, max_steal=.7):
    """Expand grace notes on a timeline.

    Expand durations of grace notes according to their specifications,
    or according to the default settings specified using the keywords.
    The onsets/offsets of the grace notes and surrounding notes are
    set accordingly. Multiple contiguous grace notes inside a voice
    are expanded sequentially.

    The specified timeline object will be modified in place.

    Parameters
    ----------
    timeline : Timeline
        The timeline on which to expand the grace notes
    default_type : str, optional. Default: 'appoggiatura'
        The type of grace note, if no type is specified in the grace
        note itself. Possibilites are: {'appoggiatura',
        'acciaccatura'}.
    min_steal : float, optional
        The minimal proportion of the note to steal wherever no
        proportion is speficied in the grace notes themselves.
    max_steal : float, optional
        The maximal proportion of the note to steal wherever no
        proportion is speficied in the grace notes themselves.
    
    """

    assert default_type in (u'appoggiatura', u'acciaccatura')
    assert 0 < min_steal <= max_steal
    assert min_steal <= max_steal < 1.0

    def n_notes_to_steal(n_notes):
        return min_steal + (max_steal - min_steal) * 2 * (1 / (1 + np.exp(- n_notes + 1)) - .5)

    def shorten_main_notes_by(offset, notes, group_id):
        # start and duration of the main note
        old_start = notes[0].start
        n_dur = np.min([n.duration for n in notes])
        offset = min(n_dur * .5, offset)
        new_start_t = old_start.t + offset
        for i, n in enumerate(notes):
            old_start.starting_objects[Note].remove(n)
            timeline.add_starting_object(new_start_t, n)
            n.appoggiatura_group_id = group_id
            n.appoggiatura_duration = offset / float(n_dur)
        return new_start_t

    def shorten_prev_notes_by(offset, notes, group_id):
        old_end = notes[0].end
        n_dur = notes[0].duration
        offset = min(n_dur * .5, offset)
        new_end_t = old_end.t - offset

        for n in notes:
            old_end.ending_objects[Note].remove(n)
            timeline.add_ending_object(new_end_t, n)
            n.acciaccatura_group_id = group_id
            n.acciaccatura_duration = offset / float(n_dur)

        return new_end_t

    def set_acciaccatura_times(notes, start_t, group_id):
        N = len(notes)
        end_t = notes[0].start.t
        times = np.linspace(start_t, end_t, N + 1, endpoint=True)
        for i, n in enumerate(notes):
            n.start.starting_objects[Note].remove(n)
            timeline.add_starting_object(times[i], n)
            n.end.ending_objects[Note].remove(n)
            timeline.add_ending_object(times[i + 1], n)
            n.acciaccatura_group_id = group_id
            n.acciaccatura_idx = i
            n.acciaccatura_size = N

    def set_appoggiatura_times(notes, end_t, group_id):
        N = len(notes)
        start_t = notes[0].start.t
        times = np.linspace(start_t, end_t, N + 1, endpoint=True)
        for i, n in enumerate(notes):
            n.start.starting_objects[type(n)].remove(n)
            timeline.add_starting_object(times[i], n)
            n.end.ending_objects[type(n)].remove(n)
            timeline.add_ending_object(times[i + 1], n)
            n.appoggiatura_group_id = group_id
            n.appoggiatura_idx = i
            n.appoggiatura_size = N

    time_grouped_gns = partition(operator.attrgetter('start.t'),
                                 timeline.iter_all(GraceNote))
    times = sorted(time_grouped_gns.keys())

    group_counter = 0
    for t in times:

        voice_grouped_gns = partition(operator.attrgetter('voice'),
                                      time_grouped_gns[t])

        for voice, gn_group in voice_grouped_gns.items():

            for n in gn_group:
                if n.grace_type == 'grace':
                    n.grace_type = default_type

            type_grouped_gns = partition(operator.attrgetter('grace_type'),
                                         gn_group)

            for gtype, type_group in type_grouped_gns.items():
                total_steal_old = n_notes_to_steal(len(type_group))
                total_steal = np.sum([n.duration_from_symbolic for n
                                      in type_group])
                main_notes = [m for m in type_group[0].simultaneous_notes_in_voice
                              if not isinstance(m, GraceNote)]

                if len(main_notes) > 0:
                    total_steal = min(
                        main_notes[0].duration / 2., total_steal)

                if gtype == 'appoggiatura':
                    total_steal = np.sum([n.duration_from_symbolic for n
                                          in type_group])
                    if len(main_notes) > 0:
                        new_onset = shorten_main_notes_by(
                            total_steal, main_notes, group_counter)
                        set_appoggiatura_times(
                            type_group, new_onset, group_counter)
                        group_counter += 1

                elif gtype == 'acciaccatura':
                    prev_notes = [m for m in type_group[0].previous_notes_in_voice
                                  if m.grace_type is None]
                    if len(prev_notes) > 0:
                        new_offset = shorten_prev_notes_by(
                            total_steal, prev_notes, group_counter)
                        set_acciaccatura_times(
                            type_group, new_offset, group_counter)
                        group_counter += 1


def iter_parts(partlist):
    """Iterate over all Part instances in partlist, which is a list of
    either Part or PartGroup instances. PartGroup instances contain
    one or more parts or further partgroups, and are traversed in a
    depth-first fashion.

    This function is designed to take the result of
    :func:`partitura.load_midi` and :func:`partitura.load_musicxml` as
    input.

    Parameters
    ----------
    partlist : list, Part, or PartGroup
        A :class:`partitura.score.Part` object,
        :class:`partitura.score.PartGroup` or a list of these

    Returns
    -------
    iterator
        Iterator over Part instances in `partlist`

    """

    if not isinstance(partlist, (list, tuple, set)):
        partlist = [partlist]

    for el in partlist:
        if isinstance(el, Part):
            yield el
        else:
            for eel in iter_parts(el.children):
                yield eel


def _repeats_to_start_end(repeats, first, last):
    """Return pairs of (start, end) TimePoints corresponding to the start and
    end times of each Repeat object. If any of the start or end attributes
    are None, replace it with the end/start of the preceding/succeeding
    Repeat, respectively, or `first` or `last`.

    Parameters
    ----------
    repeats : list
        list of Repeat instances, possibly with None-valued start/end
        attributes
    first : TimePoint
        The first TimePoint in the timeline
    last : TimePoint
        The last TimePoint in the timeline

    Returns
    -------
    list
        list of (start, end) TimePoints corresponding to each Repeat in
        `repeats`

    """
    t = first
    starts = []
    ends = []
    for repeat in repeats:
        starts.append(t if repeat.start is None else repeat.start)
        if repeat.end is not None:
            t = repeat.end

    t = last
    for repeat in reversed(repeats):
        ends.append(t if repeat.end is None else repeat.end)
        if repeat.start is not None:
            t = repeat.start
    ends.reverse()
    return list(zip(starts, ends))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
