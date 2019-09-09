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

TODO: example of Part/Timeline construction from scratch

TODO: explain that ontology roughly follows MusicXML schema (should be mostly compatible with MEI as well)

"""

import sys
import string
import re
from copy import copy
from textwrap import dedent
from collections import defaultdict
import logging
import operator
import itertools
from numbers import Number

import numpy as np
from scipy.interpolate import interp1d

from partitura.utils import ComparableMixin, partition, iter_subclasses, iter_current_next, sorted_dict_items, PrettyPrintTree

# the score ontology for longer scores requires a high recursion limit
# increase when needed
sys.setrecursionlimit(100000)

LOGGER = logging.getLogger(__name__)
_LABEL_DURS = {
    'long': 16,
    'breve': 8,
    'whole': 4,
    'half': 2,
    'h': 2,
    'quarter': 1,
    'q': 1,
    'eighth': 1/2,
    'e': 1/2,
    '16th': 1/4,
    '32nd': 1/8.,
    '64th': 1/16,
    '128th': 1/32,
    '256th': 1/64
}
_DOT_MULTIPLIERS = (1, 1+1/2, 1+3/4, 1+7/8)
_MIDI_BASE_CLASS = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}
_MORPHETIC_BASE_CLASS = {'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6}
_MORPHETIC_OCTAVE = {0: 32, 1: 39, 2: 46, 3: 53, 4: 60, 5: 67, 6: 74, 7: 81, 8: 89}
_ALTER_SIGNS = {None: '', 0: '', 1: '#', 2: 'x', -1: 'b', -2: 'bb'}

# def tree_prefix(level, mode=0):
#     if mode == 0:
#         0 =  ' │ '
#     '│ ├ ─ └'

def estimate_symbolic_duration(dur, div, eps=10**-3):
    """
    Given a numeric duration, a divisions value (specifiying the number of units
    per quarter note) and optionally a tolerance `eps` for numerical
    imprecisions, estimate corresponding the symbolic duration. If a matching
    symbolic duration is found, it is returned as a tuple (type, dots), where
    type is a string such as 'quarter', or '16th', and dots is an integer
    specifying the number of dots. If no matching symbolic duration is found the
    function returns None.
    
    NOTE this function does not estimate composite durations, nor
    time-modifications such as triplets.

    Parameters
    ----------
    dur: float or int
        Numeric duration value
    div: int
        Number of units per quarter note
    eps: float, optional (default: 10**-3)
        Tolerance in case of imprecise matches
    
    Examples
    --------

    >>> estimate_symbolic_duration(24, 16)
    {'type': 'quarter', 'dots': 1}
    
    The following example returns None:

    >>> estimate_symbolic_duration(23, 16)


    Returns
    -------
    dict or None
        A dictionary containing keys 'type' and 'dots' expressing the estimated
        symbolic duration or None
    """
    global _DOT_MULTIPLIERS, _LABEL_DURS

    for i, dotm in enumerate(_DOT_MULTIPLIERS):

        ddur = (dur/div)/dotm

        for k, v in _LABEL_DURS.items():

            if np.abs(ddur-v) < eps:

                return dict(type=k, dots=i)

    return None


def to_quarter_tempo(unit, tempo):
    """
    Given a string `unit` (e.g. 'q', 'q.' or 'h') and a number `tempo`, return
    the corresponding tempo in quarter notes. This is useful to convert textual
    tempo directions like h=100.
    
    Parameters
    ----------
    unit: str
        Tempo unit
    tempo: number
        Tempo value
    
    Returns
    -------
    float
        Tempo value in quarter units

    Examples
    --------
    >>> to_quarter_tempo('q', 100)
    100.0
    >>> to_quarter_tempo('h', 100)
    200.0
    >>> to_quarter_tempo('h.', 50)
    150.0
    """
    dots = unit.count('.')
    unit = unit.strip().rstrip('.')
    return float(tempo * _DOT_MULTIPLIERS[dots] * _LABEL_DURS[unit])


def format_symbolic_duration(symbolic_dur):
    """
    Create a string representation of the symbolic duration encoded in the
    dictionary `symbolic_dur`.

    Examples
    --------
    >>> format_symbolic_duration({'type': 'q', 'dots': 2})
    'q..'
    >>> format_symbolic_duration({'type': '16th'})
    '16th'
    

    Parameters
    ----------
    symbolic_dur: dict
        Dictionary with keys 'type' and 'dots'
    
    Returns
    -------
    str
        A string representation of the specified symbolic duration
    """
    if symbolic_dur is None:
        return 'unknown'
    else:
        return (symbolic_dur.get('type') or '')+'.'*symbolic_dur.get('dots', 0)


def symbolic_to_numeric_duration(symbolic_dur, divs):
    # TODO: can symbolic_dur be None?
    numdur = divs * _LABEL_DURS[symbolic_dur.get('type', None)]
    numdur *= _DOT_MULTIPLIERS[symbolic_dur.get('dots', 0)]
    numdur *= ((symbolic_dur.get('normal_notes') or 1) / 
               (symbolic_dur.get('actual_notes') or 1))
    return numdur



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
                if o is None:
                    pass
                elif isinstance(o, list):
                    o_list_new = []

                    for o_el in o:
                        if o_el in o_map:
                            o_list_new.append(o_map[o_el])
                        else:
                            LOGGER.warning(dedent(f'''reference not found in 
                            o_map: {o_el} start={o_el.start} end={o_el.end},
                            substituting None'''))
                            o_list_new.append(None)

                    setattr(self, attr, o_list_new)
                else:
                    if o in o_map:
                        o_new = o_map[o]
                    else:
                        if isinstance(o, Note):
                            m = o.start.get_prev_of_type(Measure, eq=True)[0]
                        LOGGER.warning(dedent(f'''reference not found in o_map:
                        {o} start={o.start} end={o.end}, substituting None'''))
                        o_new = None
                    setattr(self, attr, o_new)


class TimeLine(object):

    """
    The `TimeLine` class collects `TimePoint` objects in a doubly
    linked list fashion (as well as in an array).

    Parameters
    ----------
    No parameters

    Attributes
    ----------
    points : numpy array of TimePoint objects
        a numpy array of TimePoint objects.

    """

    def __init__(self):
        self.points = np.array([], dtype=TimePoint)

    def add_point(self, tp):
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

    def remove_point(self, tp):
        """
        remove `TimePoint` object `tp` from the time line
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
        """
        return the `TimePoint` object with time `t`, or None if there
        is no such object

        """
        i = np.searchsorted(self.points, TimePoint(t))
        if i < len(self.points) and self.points[i].t == t:
            return self.points[i]
        else:
            return None

    def get_or_add_point(self, t):
        """
        return the `TimePoint` object with time `t`; if there is no
        such object, create it, add it to the time line, and return
        it

        :param t: time value `t` (float)

        :returns: a TimePoint object with time `t`

        """

        tp = self.get_point(t)
        if tp is None:
            tp = TimePoint(t)
            self.add_point(tp)
        return tp

    def add_starting_object(self, t, o):
        """
        add object `o` as an object starting at time `t`

        """
        self.get_or_add_point(t).add_starting_object(o)

    def add_ending_object(self, t, o):
        """
        add object `o` as an object ending at time `t`

        """
        self.get_or_add_point(t).add_ending_object(o)

    def remove_starting_object(self, o):
        """
        remove object `o` as an object starting at time `t`. This involves:
          - removing `o` from the timeline
          - remove the note start timepoint if no starting or ending objects
            remain at that time
          - set o.start to None
        """
        if o.start:
            try:
                o.start.starting_objects[o.__class__].remove(o)
            except:
                raise Exception('Not implemented: removing an starting object that is registered by its superclass')
            if (sum(len(oo) for oo in o.end.starting_objects.values()) +
                sum(len(oo) for oo in o.end.ending_objects.values())) == 0:
                self.remove_point(o.start)
            o.start = None

    def remove_ending_object(self, o):
        """
        remove object `o` as an object ending at time `t`. This involves:
          - removing `o` from the timeline
          - remove the note ending timepoint if no starting or ending objects
            remain at that time
          - set o.end to None
        """
        if o.end:
            try:
                o.end.ending_objects[o.__class__].remove(o)
            except:
                raise Exception('Not implemented: removing an ending object that is registered by its superclass')
            if (sum(len(oo) for oo in o.end.starting_objects.values()) +
                sum(len(oo) for oo in o.end.ending_objects.values())) == 0:
                self.remove_point(o.end)
            o.end = None
        # self.get_or_add_point(t).add_ending_object(o)
        
    def get_all_starting(self, cls, start=None, end=None, include_subclasses=False):
        return self.get_all(cls, start, end, include_subclasses, mode='starting')

    def get_all_ending(self, cls, start=None, end=None, include_subclasses=False):
        return self.get_all(cls, start, end, include_subclasses, mode='ending')
    
    def get_all(self, cls, start=None, end=None, include_subclasses=False, mode='starting'):
        """
        return all objects of type `cls`

        """
        if not mode in ('starting', 'ending'):
            LOGGER.warning(f'unknown mode "{mode}", using "starting" instead')
            mode = 'starting'
        
        if start is not None:
            if not isinstance(start, TimePoint):
                start = TimePoint(start)

            start_idx = np.searchsorted(
                self.points, start, side='left')
        else:
            start_idx = 0

        if end is not None:
            if not isinstance(end, TimePoint):
                end = TimePoint(end)
            end_idx = np.searchsorted(self.points, end, side='left')
        else:
            end_idx = len(self.points)

        r = []
        if mode == 'ending':
            for tp in self.points[start_idx: end_idx]:
                r.extend(tp.get_ending_objects_of_type(cls, include_subclasses))
        else:
            for tp in self.points[start_idx: end_idx]:
                r.extend(tp.get_starting_objects_of_type(cls, include_subclasses))
        return r

    @property
    def last_point(self): 
        return self.points[-1] if len(self.points) > 0 else None

    @property
    def first_point(self): 
        return self.points[0] if len(self.points) > 0 else None

    def get_all_ongoing_objects(self, t):
        if not isinstance(t, TimePoint):
            t = TimePoint(t)
        t_idx = np.searchsorted(self.points, t, side='left')
        ongoing = set()
        for tp in self.points[: t_idx]:
            for starting in tp.starting_objects.values():
                for o in starting:
                    ongoing.add(o)
            for ending in tp.ending_objects.values():
                for o in ending:
                    ongoing.remove(o)
        return ongoing
        
    def test(self):
        s = set()
        for tp in self.points:
            for k, oo in tp.starting_objects.items():
                for o in oo:
                    s.add(o)
            for k, oo in tp.ending_objects.items():
                for o in oo:
                    assert o in s
                    s.remove(o)
        LOGGER.info('Timeline is OK')
        return True
    
class TimePoint(ComparableMixin, ReplaceRefMixin):

    """
    A TimePoint represents an instant in Time.

    Parameters
    ----------
    t : number
        Time point of some event in/element of the score, where the unit
        of a time point is the <divisions> as defined in the musicxml file,
        more precisely in the corresponding score part.
        Represents the absolute time of the time point, also used
        for ordering TimePoint objects w.r.t. each other.

    label : str, optional. Default: ''

    Attributes
    ----------
    t : number

    label : str

    starting_objects : dictionary
        a dictionary where the musical objects starting at this
        time are grouped by class.

    ending_objects : dictionary
        a dictionary where the musical objects ending at this
        time are grouped by class.

    * `prev`: the preceding time instant (or None if there is none)

    * `next`: the succeding time instant (or None if there is none)

    The `TimeLine` class stores sorted TimePoint objects in an array
    under TimeLine.points, as well as doubly linked (through the
    `prev` and `next` attributes). The `TimeLine` class also has
    functionality to add, and remove TimePoints.

    """

    def __init__(self, t):
        self.t = t
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
        return f'Timepoint {self.t}'

    def add_starting_object(self, obj):
        """
        add object `obj` to the list of starting objects
        """
        obj.start = self
        self.starting_objects[type(obj)].append(obj)

    def add_ending_object(self, obj):
        """
        add object `obj` to the list of ending objects
        """
        obj.end = self
        self.ending_objects[type(obj)].append(obj)

    def get_starting_objects_of_type(self, otype, include_subclasses=False):
        """
        return all objects of type `otype` that start at this time point
        """
        if include_subclasses:
            return self.starting_objects[otype] + \
                list(itertools.chain(*(self.starting_objects[subcls]
                                       for subcls in iter_subclasses(otype))))
        else:
            return self.starting_objects[otype]

    def get_ending_objects_of_type(self, otype, include_subclasses=False):
        """
        return all objects of type `otype` that end at this time point
        """
        if include_subclasses:
            return self.ending_objects[otype] + \
                list(itertools.chain(*(self.ending_objects[subcls]
                                       for subcls in iter_subclasses(otype))))
        else:
            return self.ending_objects[otype]

    def get_prev_of_type(self, otype, eq=False):
        """
        return the object(s) of type `otype` that start at the latest
        time before this time point (or at this time point, if `eq` is True)
        """
        if eq:
            value = self.get_starting_objects_of_type(otype)
            if len(value) > 0:
                return value[:]
        return self._get_prev_of_type(otype)

    def _get_prev_of_type(self, otype, eq=False):
        if self.prev is None:
            return []
        else:
            r = self.prev.get_starting_objects_of_type(otype)
            if r != []:
                return r[:]
            else:
                return self.prev._get_prev_of_type(otype)

    def get_next_of_type(self, otype, eq=False):
        """
        return the object(s) of type `otype` that start at the earliest
        time after this time point (or at this time point, if `eq` is True)
        """
        if eq:
            value = self.get_starting_objects_of_type(otype)
            if len(value) > 0:
                return value[:]
        return self._get_next_of_type(otype)

    def _get_next_of_type(self, otype, eq=False):
        if self.next is None:
            return []
        else:
            r = self.next.get_starting_objects_of_type(otype)
            if r != []:
                return r[:]
            else:
                return self.next._get_next_of_type(otype)

    def _cmpkey(self):
        """
        This method returns the value to be compared
        (code for that is in the ComparableMixin class)

        """
        return self.t

    # __hash__ = _cmpkey # this is related to the ComparableMixin

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

    """
    class that represents objects that (may?) have a start and ending
    point.
    Used as super-class for classes representing different types of
    objects in a (printed) score.
    """

    def __init__(self):
        super().__init__()
        self.start = None
        self.end = None
        # intermediate time points
        # self.intermediate = []
        # self._ref_attrs = ['start', 'end']
        

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
        self._ref_attrs.extend(['start_note', 'end_note'])

    def __str__(self):
        # return 'slur at voice {0} (ends at {1})'.format(self.voice, self.end and self.end.t)
        start = '' if self.start_note is None else f'start={self.start_note.id}'
        end = '' if self.end_note is None else f'end={self.end_note.id}'
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
        return f'Fermata ref={self.ref}'


class Ending(TimedObject):

    """
    Class that represents one part of a 1---2--- type ending of a
    musical passage (a.k.a Volta brackets).
    """

    def __init__(self, number):
        super().__init__()
        self.number = number

    def __str__(self):
        return 'Ending (from {0} to {1})'.format(self.start.t, self.end.t)


class Measure(TimedObject):

    """

    Attributes
    ----------
    number : number
        the number of the measure. (directly taken from musicxml file?)

    page :

    system :

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

    def get_measure_duration(self, quarter=False):
        """
        Return the measure duration, either in beats or in quarter note units.

        Parameters
        ----------
        quarter: bool (optional, default: False)
            If True, return the measure duration in quarter note units, otherwise in beat units

        Returns
        -------
        float
            The measure duration
        """

        # TODO: support mid-measure time sig and division changes
        divs = self.start.get_prev_of_type(Divisions, eq=True)
        ts = self.start.get_prev_of_type(TimeSignature, eq=True)
        assert len(divs) > 0
        assert len(ts) > 0
        measure_dur = self.end.t - self.start.t
        beats = ts[0].beats
        beat_type = ts[0].beat_type
        div = float(divs[0].divs)

        if quarter:
            return measure_dur / div
        else:
            return beat_type * measure_dur / (4. * div)

    @property
    def incomplete(self):
        """
        Returns True if the duration of the measure is less than the expected
        duration (as computed based on the current divisions and time
        signature), and False otherwise.

        WARNING: this property does not work reliably to detect
        incomplete measures in the middle of the piece

        NOTE: this is probably because of the way incomplete measures are dealt
        with in the musicxml parser. When a score part has an incomplete measure
        where the corresponding measure in some other part is complete, the
        duration of the incomplete measure is adjusted to that it is complete
        (otherwise the times would be misaligned after the incomplete measure).

        Returns
        -------
        bool
        """

        assert self.start.next is not None, LOGGER.error(
            'Part is empty')
        divs = self.start.next.get_prev_of_type(Divisions)
        ts = self.start.next.get_prev_of_type(TimeSignature)
        # nextm = self.start.get_next_of_type(Measure)

        invalid = False
        if len(divs) == 0:
            LOGGER.warning('Part specifies no divisions')
            invalid = True
        if len(ts) == 0:
            LOGGER.warning('Part specifies no time signatures')
            invalid = True
        # if len(nextm) == 0:
        #     LOGGER.warning('Part has just one measure')
        #     # invalid = True

        if invalid:
            LOGGER.warning(
                'could not be determine if meaure is incomplete, assuming complete')
            return False

        # measure_dur = nextm[0].start.t - self.start.t
        measure_dur = self.end.t - self.start.t
        beats = ts[0].beats
        beat_type = ts[0].beat_type
        div = float(divs[0].divs)

        # this will return a boolean, so either True or False
        # return beat_type * measure_dur / (4 * div * beats) % 1 > 0
        return beat_type*measure_dur < 4*div*beats

    # upbeat is deprecated in favor of incomplete
    @property
    def upbeat(self):
        return self.incomplete

    
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


class Divisions(TimedObject):

    """
    represents <divisions>xxx</divisions> that are used inside a measure
    to set the length of a quarter note (xxx here is the value for a quarter
    note, e.g. 256). This element usually is present in the first measure
    of each score part.
    """

    def __init__(self, divs):
        super().__init__()
        self.divs = divs

    def __str__(self):
        return 'Divisions: quarter={0}'.format(self.divs)


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
    """

    Parameters
    ----------
    fifths : number
        Number of sharps (positive) or flats (negative)

    mode : str
        Mode of the key, either 'major' or 'minor'

    """

    names = ['C','G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F']
    
    def __init__(self, fifths, mode):
        super().__init__()
        self.fifths = fifths
        self.mode = mode
        
    @property
    def name(self):
        """
        A human-readable representation of the key, such as "C#" for C sharp major,
        or "Ebm" for E flat minor. Compatible with the key signature names used
        by mido.

        Returns
        -------
        str
            Human-readable representation of the key
        """
        
        if self.mode == 'minor':
            o = 3
            m = 'm'
        else:
            o = 0
            m = ''

        return self.names[(len(self.names) + self.fifths + o) % len(self.names)] + m
    
    def __str__(self):
        return f'Key signature: fifths={self.fifths}, mode={self.mode} ({self.name})'


class Transposition(TimedObject):

    """
    represents a <transpose> tag that tells how to change all (following)
    pitches of that part to put it to concert pitch (i.e. sounding pitch).

    Parameters
    ----------
    diatonic : number

    chromatic : number
        the number of semi-tone steps to add or subtract to the pitch to
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
    """
    Represents the common aspects of notes and rests (and in the future unpitched notes)

    Parameters
    ----------
    voice : integer, optional (default: None)

    id : integer, optional (default: None)

    """
    def __init__(self, id=None, voice=None, staff=None, symbolic_duration=None, articulations={}):
        super().__init__()
        self.voice = voice
        self.id = id
        self.staff = staff
        # self.symbolic_duration = symbolic_duration
        self._sym_dur = symbolic_duration
        self.articulations = articulations

        # these attributes are set after the instance is constructed
        self.fermata = None
        self.tie_prev = None
        self.tie_next = None
        self.slur_stops = []
        self.slur_starts = []
        
        self._ref_attrs.extend(['tie_prev', 'tie_next', 'slur_stops', 'slur_starts'])

    @property
    def symbolic_duration(self):
        eps = 10**-3
        if self._sym_dur is None:
            divs = next(iter(self.start.get_prev_of_type(Divisions, eq=True)), None)
            if divs is not None:
                return estimate_symbolic_duration(self.end.t - self.start.t, divs.divs, eps)
            else:
                None
        else:
            return self._sym_dur
    
    @property
    def duration(self):
        """
        The duration of the note in divisions

        Returns
        -------
        number
        """

        try:
            return self.end.t - self.start.t
        except:
            LOGGER.warn('no end time found for note')
            return 0

    @property
    def end_tied(self):
        """
        The `Timepoint` corresponding to the end of the note, or---when this note
        belongs to a group of tied notes---the end of the last note in the
        group.
        
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
        """
        Time difference of the start of the note to the end of the note, or---when 
        this note belongs to a group of tied notes---the end of the last note in the group.
        
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
            divs = self.start.get_prev_of_type(Divisions, eq=True)
            if len(divs) == 0:
                div = 1
            else:
                div = divs[0].divs
            return symbolic_to_numeric_duration(self.symbolic_duration, div)
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
        # s = f'{type(self).__name__}: id={self.id} voice={self.voice} staff={self.staff} type="{format_symbolic_duration(self.symbolic_duration)}"'
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
                         f'pitch={self.step}{self.alter_sign}{self.octave}'))

    @property
    def midi_pitch(self):
        """
        the midi pitch value of the note (MIDI note number).
        C4 (middle C, in german: c') is note number 60.

        Returns
        -------
        integer
            the note's pitch as MIDI note number.
        """
        return ((self.octave + 1) * 12
                + _MIDI_BASE_CLASS[self.step.lower()]
                + (self.alter or 0))

    @property
    def morphetic_pitch(self):
        """
        the morphetic value of the note, i.e. a single integer.
        It corresponds to the (vertical) position of the note in
        the barline system.

        Returns
        -------
        integer
        """
        return (_MORPHETIC_OCTAVE[self.octave] +
                _MORPHETIC_BASE_CLASS[self.step.lower()])

    @property
    def alter_sign(self):
        """
        the alteration of the note

        Returns
        -------
        str
        """
        return _ALTER_SIGNS[self.alter]


    @property
    def previous_notes_in_voice(self):
        n = self
        while True:
            nn = n.start.get_prev_of_type(Note)
            if nn == []:
                return nn
            else:
                voice_notes = [m for m in nn if m.voice == self.voice]
                if len(voice_notes) > 0:
                    return voice_notes
                n = nn[0]

    @property
    def simultaneous_notes_in_voice(self):
        return [m for m in self.start.starting_objects[Note]
                if m.voice == self.voice and m != self]

    @property
    def next_notes_in_voice(self):
        n = self
        while True:
            nn = n.start.get_next_of_type(Note)
            if nn == []:
                return nn
            else:
                voice_notes = [m for m in nn if m.voice == self.voice]
                if len(voice_notes) > 0:
                    return voice_notes
                n = nn[0]


class Rest(GenericNote):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GraceNote(Note):
    def __init__(self, grace_type, *args, steal_proportion=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grace_type = grace_type
        self.steal_proportion = steal_proportion


class PartGroup(object):
    """
    Represents a <part-group ...> </...> where instruments are grouped.
    Note that a part grouped is "started" and "stopped" with according
    attributes inside the respective elements.

    Parameters
    ----------

    grouping_symbol : str or None, optional
        the symbol used for grouping instruments, a <group-symbol> element,
        possibilites are:

        - 'brace' (opening curly brace, should group 2 same instruments,
                   e.g. 2 horns,  or left + right hand on piano)
        - 'square' (opening square bracket, should have same function as
                    the brace.)
        - 'bracket' (opening square bracket, should group instruments
                     of the same category, such as all woodwinds.)

        Note that there is supposed to be a hierarchy between these,
        like this: a bracket is supposed to embrace one ore multiple
        braces or squares.

    Attributes
    ----------

    grouping_symbol : str OR None

    children : list of PartGroup objects

    parent :

    number :

    parts : list of Part objects
        a list of all Part objects in this PartGroup
    """

    def __init__(self, grouping_symbol=None, name=None):
        self.grouping_symbol = grouping_symbol
        self.children = []
        self.name = name
        self.parent = None
        self.number = None

    @property
    def parts(self):
        return get_all_parts(self.children)

    def pretty(self, l=0):
        if self.name is not None:
            name_str = ' / {0}'.format(self.name)
        else:
            name_str = ''
        s = ['    ' * l + '{0}{1}'.format(self.grouping_symbol, name_str)]
        for ch in self.children:
            s.append(ch.pretty(l + 1))
        return '\n'.join(s)


class ScoreVariant(object):
    """
    """
    
    def __init__(self, start_time=0):
        self.t_unfold = start_time
        self.segments = []

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
        return f'Segment: {self.segment_times}'
    
    def clone(self):
        """
        Return a clone of the ScoreVariant
        """
        clone = ScoreVariant(self.t_unfold)
        clone.segments = self.segments[:]
        return clone

    def create_variant_timeline(self):
        timeline = TimeLine()

        for start, end, offset in self.segments:
            delta = offset - start.t
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
                    # don't repeat divisions if it hasn't changed
                    elif isinstance(o, Divisions):
                        prev = next(iter(tp_new.get_prev_of_type(Divisions)), None)
                        if prev is not None and o.divs == prev.divs:
                            continue
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

    """
    Represents a whole score part, e.g. all notes of one single instrument
    or 2 instruments written in the same staff.
    Note that there may be more than one staff per score part; vice versa,
    in the printed score, there may be more than one score part's notes
    in the same staff (such as two flutes in one staff, etc).

    Parameters
    ----------
    part_id : str
        the id of the part (<score-part id="P1">), will look
        like 'P1' for part 1, etc.

    tl : TimeLine object OR None, optional

    Attributes
    ----------
    part_id : str

    timeline : TimeLine object

    part_name : str
        as taken from the musicxml file

    part_abbreviation : str
        as taken from the musicxml file

    notes :

    notes_unfolded :

    beat_map : scipy interpolate interp1d object
        the timeline on a beat basis, i.e. defined on the currently
        present time signature's denominator (may vary throughout the score).
        Each timepoint of the timeline is expressed as a (fraction) of
        a beat number.

    quarter_map : scipy interpolate interp1d object
        the timeline on a quarter note basis. Each timepoint of
        the timeline is be expressed as a (fraction of) a quarter
        note.
    """

    def __init__(self, part_id, timeline=None):
        self.part_id = part_id
        self.timeline = timeline or TimeLine()
        self.parent = None
        self.part_name = None
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


    def add(self, start, obj, end=None):
        """
        Add an object to the timeline.
        
        Parameters
        ----------
        start: int
            Start time of the object 
        obj: TimedObject
            Object to be added to the timeline
        end: int, optional
            End time of the object. When end is None no end time is registered
            for the object. Defaults to None.
        
        """
        
        self.timeline.add_starting_object(start, obj)

        if end is not None:

            self.timeline.add_ending_object(end, obj)


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

            
    def make_score_variants(self):
        """
        Create a list of ScoreVariant objects, each representing a
        distinct way to unfold the score, based on the repeat
        structure.

        Parameters
        ----------


        Returns
        -------

        Note
        ----

        This function does not currently support nested repeats, such as in case
        45d of the MusicXML Test Suite.

        """

        if len(self.timeline.get_all(DaCapo) +
               self.timeline.get_all(Fine)) > 0:
            LOGGER.warning(('Generation of repeat structures involving da '
                            'capo/fine/coda/segno directions is not '
                            'supported yet'))

        repeats = self.timeline.get_all(Repeat)
        # repeats may not have start or end times. `_repeats_to_start_end`
        # returns the start/end paisr for each repeat, making educated guesses
        # when these are missing.
        repeat_start_ends = _repeats_to_start_end(repeats,
                                                 self.timeline.first_point,
                                                 self.timeline.last_point)

        # check for nestings and raise if necessary
        if any(n < c for c, n in iter_current_next(repeat_start_ends)):
            raise NotImplementedError('Nested endings are currently not supported')
        
        # t_score is used to keep the time in the score
        t_score = self.timeline.first_point
        svs = [ScoreVariant()]
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
                endings1 = rep_end.get_ending_objects_of_type(Ending)
                # is there an ending?
                if len(endings1) > 0:
                    # yes
                    ending1 = endings1[0]

                    # add the first occurrence of the repeat
                    sv.add_segment(rep_start, ending1.start)

                    endings2 = rep_end.get_starting_objects_of_type(Ending)
                    if len(endings2) > 0:
                        ending2 = endings2[0]
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
        if t_score < self.timeline.last_point:
            # no, append the interval from the current score
            # position to the end of the piece
            for sv in svs:
                sv.add_segment(t_score, self.timeline.last_point)

        return svs

    def test_timeline(self):
        """
        Test if all ending objects have occurred as starting object as
        well.
        """
        return self.timeline.test()

    def iter_unfolded_timelines(self):
        for sv in self.make_score_variants():
            yield sv.create_variant_timeline()

    def unfold_timeline_maximal(self):
        """
        Return the "maximally" unfolded timeline, that is, a copy of the timeline
        where all segments marked with repeat signs are included twice.
        
        Returns
        -------
        TimeLine
            The unfolded TimeLine
        """
        
        sv = self.make_score_variants()[-1]
        return sv.create_variant_timeline()

    def remove_grace_notes(self):
        for point in self.timeline.points:
            point.starting_objects[Note] = [n for n in point.starting_objects[Note]
                                            if n.grace_type is None]
            point.ending_objects[Note] = [n for n in point.ending_objects[Note]
                                          if n.grace_type is None]

    def expand_grace_notes(self, default_type='appoggiatura', min_steal=.05, max_steal=.7):
        """
        Expand durations of grace notes according to their
        specifications, or according to the default settings specified
        using the keywords. The onsets/offsets of the grace notes and
        surrounding notes are set accordingly. Multiple contiguous
        grace notes inside a voice are expanded sequentially.

        This function modifies the `points` attribute.

        Parameters
        ----------
        default_type : str, optional. Default: 'appoggiatura'
            The type of grace note, if no type is specified in the grace note
            itself. Possibilites are: {'appoggiatura', 'acciaccatura'}.

        min_steal : float, optional
            The minimal proportion of the note to steal wherever no proportion
            is speficied in the grace notes themselves.

        max_steal : float, optional
            The maximal proportion of the note to steal wherever no proportion
            is speficied in the grace notes themselves.
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
                self.timeline.add_starting_object(new_start_t, n)
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
                self.timeline.add_ending_object(new_end_t, n)
                n.acciaccatura_group_id = group_id
                n.acciaccatura_duration = offset / float(n_dur)

            return new_end_t

        def set_acciaccatura_times(notes, start_t, group_id):
            N = len(notes)
            end_t = notes[0].start.t
            times = np.linspace(start_t, end_t, N + 1, endpoint=True)
            for i, n in enumerate(notes):
                n.start.starting_objects[Note].remove(n)
                self.timeline.add_starting_object(times[i], n)
                n.end.ending_objects[Note].remove(n)
                self.timeline.add_ending_object(times[i + 1], n)
                n.acciaccatura_group_id = group_id
                n.acciaccatura_idx = i
                n.acciaccatura_size = N

        def set_appoggiatura_times(notes, end_t, group_id):
            N = len(notes)
            start_t = notes[0].start.t
            times = np.linspace(start_t, end_t, N + 1, endpoint=True)
            for i, n in enumerate(notes):
                n.start.starting_objects[type(n)].remove(n)
                self.timeline.add_starting_object(times[i], n)
                n.end.ending_objects[type(n)].remove(n)
                self.timeline.add_ending_object(times[i + 1], n)
                n.appoggiatura_group_id = group_id
                n.appoggiatura_idx = i
                n.appoggiatura_size = N

        grace_notes = self.list_all(GraceNote)
        time_grouped_gns = partition(
            operator.attrgetter('start.t'), grace_notes)
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
                        # print('    prev: {}'.format(len(prev_notes)))
                        if len(prev_notes) > 0:
                            new_offset = shorten_prev_notes_by(
                                total_steal, prev_notes, group_counter)
                            set_acciaccatura_times(
                                type_group, new_offset, group_counter)
                            group_counter += 1

    def _pp(self, tree):
        result = ['{}Part: name="{}" id="{}"'
                  .format(tree, self.part_name, self.part_id)]
        # stack.append(' │ ')
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
        return self._pp(PrettyPrintTree())
    
    @property
    def divisions_map(self):
        divs = np.array([(divs.start.t, divs.divs) for divs in self.list_all(Divisions)])
        if len(divs) == 0:
            # warn assumption
            divs = np.array([(self.timeline.first_point.t, 1),
                             (self.timeline.last_point.t, 1)])
        elif divs[0, 0] > self.timeline.first_point.t:
            divs = np.vstack(((self.timeline.first_point.t, divs[0, 1]),
                              divs))
        divs = np.vstack((divs,
                          (self.timeline.last_point.t, divs[-1, 1])))
        return interp1d(divs[:, 0], divs[:, 1], kind='previous')


    @property
    def time_signature_map(self):
        tss = np.array([(ts.start.t, ts.beats, ts.beat_type)
                        for ts in self.list_all(TimeSignature)])
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


    def _get_beat_map(self, quarter=False, default_div=1, default_den=4):
        """
        This returns an interpolator that will accept as input timestamps
        in divisions and returns these timestamps' beatnumbers. If the flag
        `quarter` is used, these beatnumbers will refer to quarter note steps.

        Parameters
        ----------
        quarter : boolean, optional (default: False)
            If True return a function that outputs times in quarter units,
            otherwise return a function that outputs times in beat units.
        
        default_div : int, optional (default: 1)
            Divisions to use wherever there are none specified in the timeline
            itself.

        default_den : int, optional (default: 4)
            Denominator to use wherever there is no time signature specified in
            the timeline itself.

        Returns
        -------
        scipy interpolate interp1d object
        """

        if len(self.timeline.points) == 0:
            return None
        elif len(self.timeline.points) == 1:
            return lambda x: np.zeros(len(x))

        try:
            first_measure = self.timeline.points[
                0].get_starting_objects_of_type(Measure)[0]
            if first_measure.incomplete:
                offset = -first_measure.get_measure_duration(quarter=quarter)
            else:
                offset = 0
        except IndexError:
            offset = 0

        divs = np.array(
            [(x.start.t, x.divs) for x in
             self.timeline.get_all(Divisions)], dtype=np.int)

        dens = np.array(
            [(x.start.t, np.log2(x.beat_type)) for x in
             self.timeline.get_all(TimeSignature)], dtype=np.int)

        if divs.shape[0] == 0:
            LOGGER.warning(("No Divisions found in Part, "
                            "assuming divisions = {0}").format(default_div))
            divs = np.array(((0, default_div),), dtype=np.int)

        if dens.shape[0] == 0:
            LOGGER.warning(("No TimeSignature found in Part, "
                            "assuming denominator = {0}").format(default_den))
            dens = np.array(((0, np.log2(default_den)),), dtype=np.int)

        # remove lines unnecessary for linear interpolation
        didx = np.r_[0, np.where(np.diff(divs[:, 1]) != 0)[0] + 1]
        divs = divs[didx]

        # remove lines unnecessary for linear interpolation
        didx = np.r_[0, np.where(np.diff(dens[:, 1]) != 0)[0] + 1]
        dens = dens[didx]

        start = self.timeline.points[0].t
        end = self.timeline.points[-1].t

        if divs[-1, 0] < end:
            divs = np.vstack((divs, (end, divs[-1, 1])))

        if dens[-1, 0] < end:
            dens = np.vstack((dens, (end, dens[-1, 1])))

        if divs[0, 0] > start:
            divs = np.vstack(((start, divs[0, 1]), divs))

        if dens[0, 0] > start:
            dens = np.vstack(((start, dens[0, 1]), dens))

        if quarter:
            dens[:, 1] = 2 # i.e. np.log2(4)

        # integrate second column, where first column is time:
        # new_divs = np.cumsum(np.diff(divs[:, 0]) * divs[:-1, 1])
        new_divs = np.cumsum(np.diff(divs[:, 0]) / divs[:-1, 1])
        
        divs = divs.astype(np.float)
        divs[1:, 1] = new_divs
        divs[0, 1] = divs[0, 0]

        # at this point divs[:, 0] is a list of musicxml div times
        # and divs[:, 1] is a list of corresponding quarter note times

        # interpolation object to map div times to quarter times:
        div_intp = interp1d(divs[:, 0], divs[:, 1])

        dens = dens.astype(np.float)
        # change dens[:, 0] from div to quarter times
        dens[:, 0] = div_intp(dens[:, 0])
        # change dens[:, 1] back from log2(beat_type) to beat_type and divide by
        # 4; Here take the reciprocal (4 / 2**dens[:, 1]) since in divid_outside_cumsum we will be
        # dividing rather than multiplying:
        dens[:, 1] = 4 / 2**dens[:, 1]

        # dens_new = np.cumsum(np.diff(dens[:, 0]) * dens[:-1, 1])
        dens_new = np.cumsum(np.diff(dens[:, 0]) / dens[:-1, 1])
        
        dens[1:, 1] = dens_new
        dens[0, 1] = dens[0, 0]

        den_intp = interp1d(dens[:, 0], dens[:, 1])

        if len(self.timeline.points) < 2:
            return lambda x: np.zeros(len(x))
        else:
            def f(x):
                try:
                    return den_intp(div_intp(x)) + offset
                except ValueError:
                    raise
            return f

    def get_loudness_directions(self):
        """
        Return all loudness directions

        """
        return self.list_all(LoudnessDirection, unfolded=unfolded, include_subclasses=True)

    def get_tempo_directions(self, unfolded=False):
        """
        Return all tempo directions

        """
        return self.list_all(TempoDirection, unfolded=unfolded, include_subclasses=True)

    def list_all(self, cls, unfolded=False, include_subclasses=False):
        if unfolded:
            tl = self.unfold_timeline_maximal()
        else:
            tl = self.timeline
        return tl.get_all(cls, include_subclasses=include_subclasses)

    @property
    def notes(self):
        """
        Return a list of all Note objects in the part. This list includes GraceNote
        objects but not Rest objects.

        Returns
        -------
        list
            list of Note objects
        """
        return self.list_all(Note, unfolded=False, include_subclasses=True)

    @property
    def notes_tied(self):
        """
        Return a list of all Note objects in the part that are either not tied, or
        the first note of a group of tied notes. This list includes GraceNote
        objects but not Rest objects.

        Returns
        -------
        list
            list of Note objects
        """
        notes = [note for note in self.list_all(Note, unfolded=False, include_subclasses=True)
                 if note.tie_prev is None]
        return notes

    @property
    def notes_unfolded(self):
        """
        return all note objects of the score part, after unfolding the timeline

        Returns
        -------
        list
            list of Note objects

        """
        return self.list_all(Note, unfolded=True)

    @property
    def beat_map(self):
        """
        A function that maps timeline times to beat units

        Returns
        -------
        function
            The mapping function

        """
        return self._get_beat_map()

    @property
    def quarter_map(self):
        """
        A function that maps timeline times to quarter note units

        Returns
        -------
        function
            The mapping function

        """
        return self._get_beat_map(quarter=True)

def iter_parts(partlist):
    """
    Iterate over all Part instances in partlist, which is a list of either Part
    or PartGroup instances. PartGroup instances contain one or more parts or
    further partgroups.
    
    Parameters
    ----------
    partlist: list
        Description of `partlist`
    
    Returns
    -------
    iterator
        Iterator over Part instances
    """
    for el in partlist:
        if isinstance(el, Part):
            yield el
        else:
            for eel in iter_parts(el.children):
                yield eel


def _repeats_to_start_end(repeats, first, last):
    """
    Return pairs of (start, end) TimePoints corresponding to the start and end
    times of each Repeat object. If any of the start or end attributes are None,
    replace it with the end/start of the preceding/succeeding Repeat,
    respectively, or `first` or `last`.
    
    Parameters
    ----------
    repeats: list
        list of Repeat instances, possibly with None-valued start/end attributes
    first: TimePoint
        The first TimePoint in the timeline
    last: TimePoint
        The last TimePoint in the timeline
    
    Returns
    -------
    list
        list of (start, end) TimePoints corresponding to each Repeat in `repeats`
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


def find_tie_split(start, end, divs):
    """
    Try to split the time interval from `start` to `end` into subintervals such
    that each subinterval corresponds to a symbolic duration, given a quarter
    note duration of `divs`. This can be used to split a note with a compound
    duration into a series of notes with simple duration.

    The result is returned as a list of triplets (sub_start, sub_end, sym_dur),
    one for each subinterval, where sub_start and sub_end are the start and end
    times of the subinterval and sym_dur is a dictionary representation of the
    symbolic duration, as returned by `estimate_symbolic_duration`.

    If no split can be found such that all of the subintervals correspond to a
    symbolic duration, the function returns an empty list.

    Parameters
    ----------
    start: int
        Start of the interval
    end: int
        End of the interval
    divs: int
        Duration of a quarter note
    
    Returns
    -------
    list
        The list of subintervals and their symbolic durations

    Examples
    --------

    >>> durs = find_tie_split(0, 13, 4)
    >>> for dur in durs: print(dur)
    (0, 4, {'type': 'quarter', 'dots': 0})
    (4, 8, {'type': 'quarter', 'dots': 0})
    (8, 10, {'type': 'eighth', 'dots': 0})
    (10, 13, {'type': 'eighth', 'dots': 1})

    >>> durs = find_tie_split(1, 14, 4)
    >>> for dur in durs: print(dur)
    (1, 4, {'type': 'eighth', 'dots': 1})
    (4, 8, {'type': 'quarter', 'dots': 0})
    (8, 14, {'type': 'quarter', 'dots': 1})

    Notes
    -----

    This function performs a depth first search and in case there are multiple
    solutions it does not prioritize the most musically plausible solution.
    """
    
    
    # start by trying the largest fractional units
    unit = divs*2**int(np.log2( (end-start) / divs))
    return find_tie_split_inner(start, end, divs, unit)


def find_tie_split_inner(start, end, divs, unit):
    # This is a helper function for the `find_tie_split` function
    dur = estimate_symbolic_duration(end - start, divs)
    if dur is not None:
        return [(start, end, dur)]
    else:
        t1 = unit * (start // unit + 1)
        t2 = unit * (end // unit)
        split = t1
        while split < t2:
            dur1 = find_tie_split(start, split, divs)
            dur2 = find_tie_split(split, end, divs)

            if dur1 and dur2:
                return dur1 + dur2
            elif dur1:
                dur2 = find_tie_split(split, end, divs)
                if dur2:
                    return dur1 + dur2
            elif dur2:
                dur1 = find_tie_split(start, split, divs)
                if dur1:
                    return dur1 + dur2

            split += unit
        if unit % 2 > 0:
            return []
        else:
            return find_tie_split_inner(start, end, divs, unit // 2)

# def add_measures(part):
#     # WIP
#     divs = part.list_all(Divisions)
#     ts = part.list_all(TimeSignature)
#     sorted(divs + ts, key=lambda o: o.start.t)


if __name__ == '__main__':
    import doctest
    doctest.testmod()


        # for i, (cls, objects) in enumerate(starting_items):
        #     if len(objects) > 0:
        #         if i == (M - 1):
        #             tree.last_item()
        #         else:
        #             tree.next_item()
        #         result.append('{}{}'.format(tree, cls.__name__))
        #         tree.push()
        #         N = len(objects)
        #         for j, o in enumerate(objects):
        #             if j == (N - 1):
        #                 tree.last_item()
        #             else:
        #                 tree.next_item()
        #             result.append('{}{}'.format(tree, o))
        #         tree.pop()
    
