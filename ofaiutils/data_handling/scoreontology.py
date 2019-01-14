#!/usr/bin/env python
# -*- coding: utf-8 -*-


from scipy.interpolate import interp1d
"""
This module contains an ontology to represent musical scores. A score
is defined at the highest level by a ScorePart object. This object
contains a TimeLine object, which as acts as a washing line for the
elements in a musical score such as measures, notes, slurs, words,
expressive directions. The TimeLine object contains a sequence of
TimePoint objects, which are the pegs that fix the score elements in
time. Each TimePoint object has a time value `t`, and optionally a
label. Furthermore, it contains a list of objects that start at `t`,
and another list of objects that end at `t`.

"""

import sys
import string
import re
from copy import copy
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from collections import defaultdict
import logging
import operator
import itertools
from numbers import Number

from ..utils.lang_utils import cached_property, ComparableMixin, iter_subclasses
from ..utils.container_utils import partition
# from annotation_tokenizer import parse_words # tokenizer, TokenizeException


# the score ontology for longer scores requires a high recursion limit
# increase when needed
sys.setrecursionlimit(100000)

logging.basicConfig()
LOGGER = logging.getLogger(__name__)

NON_ALPHA_NUM_PAT = re.compile(r'\W', re.UNICODE)


# this produces less rounding error than scipy.interpolate.interp1d

# def interp1d_old(x, y):
#     return InterpolatedUnivariateSpline(x, y, k=1)

# def my_interp1d(x, y):
#     def f(x_new):
#         if not hasattr(x_new, "__len__"):
#             x_new = np.array([x_new])
#         # output values
#         v = np.empty(len(x_new))
#         # insert index
#         i = np.searchsorted(x, x_new)
#         same = x[i] == x_new
#         v[same] = y[i[same]]
#         if np.sum(same) < len(x_new):
#             i = i[~same]
#             v[~same] = y[i-1] + (y[i] - y[i - 1]) * ( x_new[~same] - x[i - 1]) / (x[i] - x[i - 1])
#             # np.savetxt('/tmp/nsame.txt', np.column_stack((x_new[~same], x[i-1], x[i], y[i-1], y[i], v[~same])), fmt='%.3f')
#         return v

#     return f


def kahan_cumsum(x):
    """
    Return the cumsum of a sequence of numbers `x` using the Kahan sum algorithm
    to bound numerical error.

    Parameters
    ----------
    x: iterable over numbers
        A sequence of numbers to be cumsummed

    Returns
    -------
    ndarray: The cumsum of the elements in `x`
    """

    x = np.asarray(x)
    cumulator = np.zeros_like(x)
    compensation = 0.0

    cumulator[0] = x[0]
    for i in range(1, len(x)):
        y = x[i] - compensation
        t = cumulator[i - 1] + y
        compensation = (t - cumulator[i - 1]) - y
        cumulator[i] = t
    return cumulator


def divide_outside_cumsum(X):
    """
    this computes np.cumsum(np.diff(X[:, 0]) / X[:-1, 1]), but produces less
    rounding errors when X.dtype = int, by moving the division operation out of
    the cumsum.
    """
    diff = np.diff(X[:, 0])
    num = kahan_cumsum([diff[i] * np.prod(X[:i, 1]) * np.prod(X[i + 1:-1, 1])
                        for i in range(len(X) - 1)])
    den = np.prod(X[:-1, 1])
    return num / np.float(den)


def _symbolic_to_numeric_duration(symbolic_dur, divs):
    label_durs = {
        'long': 16,
        'breve': 8,
        'whole': 4,
        'half': 2,
        'quarter': 1,
        'eighth': 1./2,
        '16th': 1./4,
        '32nd': 1./8.,
        '64th': 1./16,
        '128th': 1./32,
        '256th': 1./64
    }
    dot_multipliers = (1, 1 + 1./2, 1 + 3./4, 1 + 7./8)
    numdur = divs * label_durs[symbolic_dur.get('type', 'quarter')]
    numdur *= dot_multipliers[symbolic_dur.get('dots', 0)]
    numdur *= float(symbolic_dur.get('normal_notes', 1)) / \
        symbolic_dur.get('actual_notes', 1)
    return numdur


def symbolic_to_numeric_duration(symbolic_durs, divs):
    numdur = 0
    for symbolic_dur in symbolic_durs:
        numdur += _symbolic_to_numeric_duration(symbolic_dur, divs)
    return numdur

# def preprocess_direction_name(l):
#     try:
#         to_remove = set(('COMMA','CARLOSCOMMENT', 'PARENOPEN', 'PARENCLOSE', 'TEMPOHINT'))
#         tokens = tokenizer.tokenize(l)
#         parts = []
#         for t in tokens:
#             if t.type in ('ROMAN_NUMBER', 'TEMPOHINT', 'PRIMO'):
#                 parts.append(t.value)
#             elif t.type in to_remove:
#                 continue
#             else:
#                 parts.append(t.type.lower())
#         return '_'.join(parts)
#     except TokenizeException as e:
#         return l.lower().replace(' ', '_')

# def preprocess_direction_fallback(l):
#     """
#     try to convert direction name into a normalized form; some
#     translation takes place to correct for common abbreviations
#     (e.g. rall. for rallentando), and OCR errors; furthermore the
#     string will be converted to lowercase and spaces are replaced by
#     underscores

#     this function is obsolete and should only be used if the ply module is not available

#     Parameters
#     ----------
#     l : str
#         a direction name

#     Returns
#     -------
#     str
#         a string containing the processed version of `l`
#     """

#     # TODO:
#     # Lento Sostenuto -> lento
#     # poco rall. -> rallentando
#     # poco ritenuto -> ritenuto
#     # pp e poco ritenuto -> ritenuto

#     # for simplicity of equiv replacements,
#     # do more normalization:
#     # lkey = ln.replace(',._-','')
#     lsl = l.strip().lower()
#     lkey = NON_ALPHA_NUM_PAT.sub(ur'', lsl)
#     # print(r, l)
#     # tr = string.ascii_lowercase + '_'
#     # delete_table = string.maketrans(tr, ' ' * len(tr))
#     # ln = l.strip().lower()
#     # lkey = ln.translate(None, delete_table)
#     equivalences = {u'dim': u'diminuendo',
#                     u'dimin': u'diminuendo',
#                     u'diminuend': u'diminuendo',
#                     u'diminuendosempre': u'diminuendo',
#                     u'dirn': u'diminuendo',  # OCR errors
#                     u'cresc': u'crescendo',
#                     u'cre': u'crescendo',
#                     u'ten': u'tenuto',
#                     u'cr': u'crescendo',
#                     u'rall': u'rallentando',
#                     u'espress': u'espressivo',
#                     u'pocoritenuto': u'ritenuto',
#                     u'pocoriten': u'ritenuto',
#                     u'pocorubato': u'ritardando',
#                     u'pocorall': u'rallentando',
#                     u'pocorallentando': u'rallentando',
#                     u'pizz': u'pizzicato',
#                     u'atenepo': u'a_tempo',
#                     u'rallentandomolto': u'rallentando',
#                     u'appasionato': u'appassionato',
#                     u'legatissizno': u'legatissimo',
#                     u'rallent': u'rallentando',
#                     u'rallent': u'rallentando',
#                     u'rit': u'ritardando',
#                     u'ritpocoapoco': u'ritardando',
#                     u'ritard': u'ritardando',
#                     u'riten': u'ritenuto',
#                     u'rinf': u'rinforzando',
#                     u'rinforz': u'rinforzando',
#                     u'smorz': u'smorzando',
#                     u'tenute': u'tenuto',
#                     u'pi\xf9_lento': u'piu_lento'
#                     }

#     # print('lkey', lkey, equivalences.get(lkey))
#     return equivalences.get(lkey, NON_ALPHA_NUM_PAT.sub(ur'_', lsl))


class TimeLine(object):

    """
    The `TimeLine` class collects `TimePoint` objects in a doubly
    linked list fashion (as well as in an array). Once all `TimePoint`
    objects have beed added, the TimeLine can be locked (that is, no
    more `TimePoint` objects can be added), in order to allow for
    caching of property values (without locking the correctness of the
    cached values cannot be guaranteed)

    Parameters
    ----------
    No parameters

    Attributes
    ----------
    points : numpy array of TimePoint objects
        a numpy array of TimePoint objects.

    locked : boolean
        if the timeline is locked, no points can be added until
        `unlock()` is called.
    """

    def __init__(self):
        self.points = np.array([], dtype=TimePoint)
        self.locked = False

    def lock(self):
        """
        lock the time line; no points can be added until `unlock` is called
        """
        self.locked = True

    def unlock(self):
        """
        unlock the time line; points can be added until `lock` is called
        """
        self.locked = False

    def link(self):
        """
        double link all points in the time line
        """
        for i in range(len(self.points) - 1):
            self.points[i].next = self.points[i + 1]
            self.points[i + 1].prev = self.points[i]

    def add_point(self, tp):
        """
        add `TimePoint` object `tp` to the time line
        """
        if self.locked:
            LOGGER.warning('Attempt to mutate locked TimeLine object')
        else:
            N = len(self.points)
            i = np.searchsorted(self.points, tp)
            if not (i < N and self.points[i].t == tp.t):
                self.points = np.insert(self.points, i, tp)
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
        N = len(self.points)
        i = np.searchsorted(self.points, TimePoint(t))
        if i < N and self.points[i].t == t:
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

    def get_all_of_type(self, cls, start=None, end=None, include_subclasses=False):
        """
        return all objects of type `cls`

        """
        if start is not None:
            if not isinstance(start, TimePoint):
                start = TimePoint(start)

            start_idx = np.searchsorted(
                self.points, start, side='left')
        else:
            start_idx = 0

        if end is not None:
            if not isinstance(end, TimePoint):
                end = TimePoint(start)
            end_idx = np.searchsorted(self.points, end, side='left')
        else:
            end_idx = len(self.points)

        r = []
        for tp in self.points[start_idx: end_idx]:
            r.extend(tp.get_starting_objects_of_type(cls, include_subclasses))

        return r


class TimePoint(ComparableMixin):

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
    functionality to add, remove, lock, and unlock the TimePoints.

    """

    def __init__(self, t, label=''):
        self.t = t
        self.label = label
        self.starting_objects = defaultdict(list)
        self.ending_objects = defaultdict(list)

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

    def __unicode__(self):
        return u'Timepoint {0}: {1}'.format(self.t, self.label)

    def __str__(self):
        return 'Timepoint {0}: {1}'.format(self.t, self.label)

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

    @cached_property
    def next(self):
        """
        return the next time point, or None if there is no such
        object; this property will be set when the object is part of a
        time line

        """
        return None

    @cached_property
    def prev(self):
        """
        return the previous time point, or None if there is no such
        object; this property will be set when the object is part of a
        time line

        """
        return None

    def _cmpkey(self):
        """
        This method returns the value to be compared
        (code for that is in the ComparableMixin class)

        """
        return self.t

    __hash__ = _cmpkey      # shorthand?


class TimedObject(object):

    """
    class that represents objects that (may?) have a start and ending
    point. TO DO: check!
    Used as super-class for classes representing different types of
    objects in a (printed) score.
    """

    def __init__(self):
        self.start = None
        self.end = None
        # intermediate time points
        self.intermediate = []


class Page(TimedObject):

    def __init__(self, nr=0):
        super(Page, self).__init__()
        self.nr = nr

    def __unicode__(self):
        return u'page {0}'.format(self.nr)

    def __str__(self):
        return 'page {0}'.format(self.nr)


class System(TimedObject):

    def __init__(self, nr=0):
        super(System, self).__init__()
        self.nr = nr

    def __unicode__(self):
        return u'system {0}'.format(self.nr)

    def __str__(self):
        return 'system {0}'.format(self.nr)


class Slur(TimedObject):

    """
    Parameters
    ----------
    voice : number
        the voice the slur corresponds to, this is given by a
        <voice>number_of_voice</voice> tag inside <note> ... </note>.
    """

    def __init__(self, voice):
        super(Slur, self).__init__()
        self.voice = voice

    def __unicode__(self):
        return u'slur at voice {0} (ends at {1})'.format(self.voice, self.end and self.end.t)

    def __str__(self):
        return 'slur at voice {0} (ends at {1})'.format(self.voice, self.end and self.end.t)


class Repeat(TimedObject):

    def __init__(self):
        super(Repeat, self).__init__()

    def __unicode__(self):
        return u'Repeat (from {0} to {1})'.format(self.start and self.start.t, self.end and self.end.t)

    def __str__(self):
        return 'Repeat (from {0} to {1})'.format(self.start and self.start.t, self.end and self.end.t)


class DaCapo(TimedObject):

    def __init__(self):
        super(DaCapo, self).__init__()

    def __unicode__(self):
        return u'Dacapo'  # (at {0} to {1})'.format(self.start.t, self.end.t)

    def __str__(self):
        return str(self.__unicode__())


class Fine(TimedObject):

    def __init__(self):
        super(Fine, self).__init__()

    def __unicode__(self):
        return u'Fine'

    def __str__(self):
        return str(self.__unicode__())


class Fermata(TimedObject):

    def __init__(self):
        super(Fermata, self).__init__()

    def __unicode__(self):
        return u'Fermata'

    def __str__(self):
        return str(self.__unicode__())


class Ending(TimedObject):

    """
    Object that represents one part of a 1---2--- type ending of a
    musical passage (aka Volta brackets).
    """

    def __init__(self, number):
        super(Ending, self).__init__()
        self.number = number

    def __unicode__(self):
        return u'Ending (from {0} to {1})'.format(self.start.t, self.end.t)

    def __str__(self):
        return str(self.__unicode__())


class Measure(TimedObject):

    """

    Attributes
    ----------
    number : number
        the number of the measure. (directly taken from musicxml file?)

    page :

    system :

    upbeat : boolean
    """

    def __init__(self):
        super(Measure, self).__init__()
        self.number = None
        self.page = None
        self.system = None

    def __unicode__(self):
        return u'measure {0} at page {1}, system {2}'.format(self.number, self.page, self.system)

    def __str__(self):
        return str(self.__unicode__())

    def get_measure_duration(self, quarter=False):
        """
        Parameters
        ----------
        quarter : ????, optional. Default: False

        Returns
        -------

        """

        assert self.start.next is not None, LOGGER.error(
            'Measure has no successor')
        divs = self.start.next.get_prev_of_type(Divisions)
        ts = self.start.next.get_prev_of_type(TimeSignature)
        nextm = self.start.get_next_of_type(Measure)
        assert len(divs) > 0
        assert len(ts) > 0
        assert len(nextm) > 0
        measure_dur = nextm[0].start.t - self.start.t
        beats = ts[0].beats
        beat_type = ts[0].beat_type
        div = float(divs[0].divs)

        if quarter:
            return measure_dur / div
        else:
            return beat_type * measure_dur / (4. * div)

    @property
    def upbeat(self):
        """Returns True if the duration of the measure
        is equal to the expected duration (based on
        divisions and time signature).

        NOTE: What does "expected duration" refer to here?

        WARNING: this property does not work reliably to detect
        incomplete measures in the middle of the piece

        Returns
        -------
        boolean

        """

        assert self.start.next is not None, LOGGER.error(
            'ScorePart is empty')
        divs = self.start.next.get_prev_of_type(Divisions)
        ts = self.start.next.get_prev_of_type(TimeSignature)
        nextm = self.start.get_next_of_type(Measure)

        invalid = False
        if len(divs) == 0:
            LOGGER.warning('ScorePart specifies no divisions')
            invalid = True
        if len(ts) == 0:
            LOGGER.warning('ScorePart specifies no time signatures')
            invalid = True
        if len(nextm) == 0:
            LOGGER.warning('ScorePart has just one measure')
            invalid = True

        if invalid:
            LOGGER.warning(
                'upbeat could not be determined properly, assuming no upbeat')
            return False

        measure_dur = nextm[0].start.t - self.start.t
        beats = ts[0].beats
        beat_type = ts[0].beat_type
        div = float(divs[0].divs)

        # this will return a boolean, so either True or False
        return beat_type * measure_dur / (4 * div * beats) % 1.0 > 0.0


class TimeSignature(TimedObject):

    """
    Parameters
    ----------
    beats :

    beat_type :
    """

    def __init__(self, beats, beat_type):
        super(TimeSignature, self).__init__()
        self.beats = beats
        self.beat_type = beat_type

    def __unicode__(self):
        return u'time signature: {0}/{1}'.format(self.beats, self.beat_type)

    def __str__(self):
        return str(self.__unicode__())


class Divisions(TimedObject):

    """
    represents <divisions>xxx</divisions> that are used inside a measure
    to set the length of a quarter note (xxx here is the value for a quarter
    note, e.g. 256). This element usually is present in the first measure
    of each score part.
    """

    def __init__(self, divs):
        super(Divisions, self).__init__()
        self.divs = divs

    def __unicode__(self):
        return u'divisions: quarter={0}'.format(self.divs)

    def __str__(self):
        return str(self.__unicode__())


class Tempo(TimedObject):

    def __init__(self, bpm):
        super(Tempo, self).__init__()
        self.bpm = bpm

    def __unicode__(self):
        return u'tempo: bpm={0}'.format(self.bpm)

    def __str__(self):
        return str(self.__unicode__())


class KeySignature(TimedObject):

    """
    Parameters
    ----------
    fifths :

    mode :
    """

    def __init__(self, fifths, mode):
        super(KeySignature, self).__init__()
        self.fifths = fifths
        self.mode = mode

    def __unicode__(self):
        return u'key signature: fifths={0}, mode={1}'.format(self.fifths, self.mode)

    def __str__(self):
        return str(self.__unicode__())


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
        super(Transposition, self).__init__()
        self.diatonic = diatonic
        self.chromatic = chromatic

    def __unicode__(self):
        return u'transposition: diatonic={0}, chromatic={1}'.format(self.diatonic, self.chromatic)

    def __str__(self):
        return str(self.__unicode__())


class Words(TimedObject):

    """
    Parameters
    ----------
    text : str
    """

    def __init__(self, text):
        super(Words, self).__init__()
        self.text = text

    # def __str__(self):
    #     return self.__unicode__().encode('utf8')

    def __unicode__(self):
        return u'{}: {}'.format(type(self).__name__, self.text)

    def __str__(self):
        return str(self.__unicode__())


class Direction(TimedObject):

    """

    """

    # labels = []
    # patterns = []
    def __init__(self, text):
        self.text = text
        self.start = None
        self.end = None

    # def __str__(self):
    #     return self.__unicode__().encode('utf8')

    def __unicode__(self):
        return u'{}: {}'.format(type(self).__name__, self.text)

    def __str__(self):
        return str(self.__unicode__())


class TempoDirection(Direction):
    pass


class DynamicTempoDirection(TempoDirection):
    def __init__(self, text):
        Direction.__init__(self, text)
        self.intermediate = []


class ConstantTempoDirection(TempoDirection):
    pass


class ResetTempoDirection(ConstantTempoDirection):
    pass


class LoudnessDirection(Direction):
    pass


class DynamicLoudnessDirection(LoudnessDirection):
    def __init__(self, text):
        Direction.__init__(self, text)
        self.intermediate = []


class ConstantLoudnessDirection(LoudnessDirection):
    pass


class ImpulsiveLoudnessDirection(LoudnessDirection):
    pass


class Note(TimedObject):

    """
    represents a note.

    Parameters
    ----------
    step : str
        the basic pitch class, like 'C', 'D', 'E', etc.

    alter: integer
        number of semi-tones to alterate the note from its basic pitch
        given by `step`.
        Note that the musicxml standard in principle allows for this to
        be a float number for microtones (micro-intonation). In Midi this
        would/could then translate to a pitch-bend.

    octave : integer
        the octave where octave 4 is the one having middle C (C4).

    voice : integer, optional. Default: None

    id : integer, optional. Default: None

    ...


    Attributes
    ----------
    previous_notes_in_voice :

    simultaneous_notes_in_voice :

    next_notes_in_voice :

    midi_pitch : integer

    morphetic_pitch :

    alter_sign :

    duration :

    """

    def __init__(self, step, alter, octave, voice=None, id=None,
                 symbolic_duration=None,
                 grace_type=None, steal_proportion=None,
                 staccato=False, fermata=False, accent=False,
                 coordinates=None, staff=None):
        super(Note, self).__init__()
        self.step = step
        if alter not in (None, 0, 1, 2, 3, -1, -2, 3):
            print('alter', step, alter, octave)
            raise Exception()
        if alter == 0:
            alter = None
        self.alter = alter
        self.octave = octave
        self.voice = voice
        self.id = id
        self.grace_type = grace_type
        self.steal_proportion = steal_proportion
        self.staccato = staccato
        self.fermata = fermata
        self.accent = accent
        self.staff = staff
        self.coordinates = coordinates
        self.symbolic_durations = []
        if symbolic_duration is not None:
            self.symbolic_durations.append(symbolic_duration)

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
        base_class = {'c': 0, 'd': 2, 'e': 4, 'f': 5,
                      'g': 7, 'a': 9, 'b': 11}[self.step.lower()] + (self.alter or 0)

        return (self.octave + 1) * 12 + base_class

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
        base_class = {'c': 0, 'd': 1, 'e': 2, 'f': 3,
                      'g': 4, 'a': 5, 'b': 6}[self.step.lower()]
        octave_number = {0: 32, 1: 39, 2: 46, 3: 53,
                         4: 60, 5: 67, 6: 74, 7: 81,
                         8: 89}[self.octave]

        return octave_number + base_class

    @property
    def alter_sign(self):
        """
        the alteration of the note

        Returns
        -------
        str
        """
        return {None: ' ', 1: '#', 2: 'x', -1: 'b', -2: 'bb'}[self.alter]

    @property
    def duration(self):
        """
        the duration of the note in divisions

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
    def duration_from_symbolic(self):
        divs = self.start.get_prev_of_type(Divisions, True)
        if len(divs) == 0:
            div = 1
        else:
            div = divs[0].divs
        # TODO: it is theoretically possible that the divisions change
        # in between tied notes. The current assumes this does not happen.
        return symbolic_to_numeric_duration(self.symbolic_durations, div)

    def __unicode__(self):
        return u'{0}{1}{2} ({8}-{9}, midi: {3}, duration: {5}, voice: {4}, id: {6}, {7})'\
            .format(self.alter_sign, self.step, self.octave,
                    self.midi_pitch, self.voice, self.duration,
                    self.id or '', self.grace_type if self.grace_type else '',
                    self.start and self.start.t, self.end and self.end.t)

    def __str__(self):
        return str(self.__unicode__())


def get_all_score_parts(constituents):
    """
    From a list whose elements are either ScorePart objects or
    PartGroup objects, return an ordered list of ScorePart objects.

    Parameters:
    -----------
    constituents : iterable
        a list of ScorePart/PartGroup objects

    Returns:
    --------
    iterable
        a list of all ScorePart objects embedded in `constituents`

    """

    return [score_part for constituent in constituents
            for score_part in
            ((constituent,) if isinstance(constituent, ScorePart)
             else get_all_score_parts(constituent.constituents))]


class PartGroup(object):

    """
    represents a <part-group ...> </...> where instruments are grouped.
    Note that a part grouped is "started" and "stopped" with according
    attributes inside the respective elements.

    Parameters
    ----------
    grouping_symbol : str OR None, optional
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

    constituents : list of PartGroup objects

    parent :

    number :

    score_parts : list of ScorePart objects
        a list of all ScorePart objects in this PartGroup
    """

    def __init__(self, grouping_symbol=None, name=None):
        self.grouping_symbol = grouping_symbol
        self.constituents = []
        self.name = name
        self.parent = None
        self.number = None

    @property
    def score_parts(self):
        return get_all_score_parts(self.constituents)

    def pprint(self, l=0):
        if self.name is not None:
            name_str = u' / {0}'.format(self.name)
        else:
            name_str = u''
        s = [u'  ' * l + u'{0}{1}'.format(self.grouping_symbol, name_str)]
        for ch in self.constituents:
            s.append(ch.pprint(l + 1))
        return u'\n'.join(s)


class ScoreVariant(object):

    def __init__(self, start_time=0):
        self.t_unfold = start_time
        self.segments = []

    def add_segment(self, start, end):
        self.segments.append((start, end, self.t_unfold))
        self.t_unfold += (end.t - start.t)

    def get_segments(self):
        """return segment (start, end, offset) information for each of
        the segments in the score variant.

        PHENICX NOTE: these numbers can be inserted directly into the
        ScoreVariantSequence table, as "ScoreStartBeat",
        "ScoreStopBeat", and "Offset", respectively

        """
        return [(s.t, e.t, 0 if i > 0 else o)
                for i, (s, e, o) in enumerate(self.segments)]

    def clone(self):
        clone = ScoreVariant(self.t_unfold)
        clone.segments = self.segments[:]
        return clone


class ScorePart(object):

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

    def __init__(self, part_id, tl=None):
        self.part_id = part_id
        self.timeline = TimeLine() if tl is None else tl
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
                yield u' '.join(chunks)
            part = part.parent

    def make_score_variants(self):
        """
        Create a list of ScoreVariant objects, each representing a
        distinct way to unfold the score, based on the repeat
        structure.

        Parameters
        ----------


        Returns
        -------

        """

        LOGGER.warning(('Generation of repeat structures involving da '
                        'capo/fine/coda/segno directions is not (properly) '
                        'implemented yet'))

        repeats = self.timeline.get_all_of_type(Repeat)

        # t_score is used to keep the time in the score
        t_score = TimePoint(0)
        # the last time instance in the piece
        end_point = self.timeline.points[-1]
        # t_unfold is used to keep the time in the score variant
        # t_unfold = 0
        # times will aggregate the triples that make up the result
        times = []
        # flag that tells... if we've reached a "da capo" sign in the
        # score
        reached_dacapo = False
        svs = [ScoreVariant()]
        # each repeat holds start and end time of a score interval to
        # be repeated
        for repeat in repeats:
            new_svs = []
            for sv in svs:
                # is the start of the repeat after our current score
                # position?
                if repeat.start > t_score:
                    # yes: add the tuple (t_score, repeat.start) to the
                    # result this is the span before the interval that is
                    # to be repeated
                    # times.append((t_score, repeat.start, t_unfold))
                    sv.add_segment(t_score, repeat.start)

                # get any "endings" (e.g. 1 / 2 volta) of the repeat
                # (there are not supposed to be more than one)
                endings = repeat.end.get_ending_objects_of_type(Ending)

                # create a new ScoreVariant for the repetition (sv
                # will be the score variant where this repeat is
                # played only once)
                new_sv = sv.clone()

                # is there an ending?
                if len(endings) > 0:
                    # yes
                    ending = endings[0]

                    # add the first occurrence of the repeat
                    sv.add_segment(repeat.start, ending.start)

                    # we are in the second iteration of the repeat, so
                    # only add the interval of the repeat up to the ending
                    # (rather than up to the end of the repeat)
                    # add the first occurrence of the repeat
                    new_sv.add_segment(repeat.start, repeat.end)
                    new_sv.add_segment(repeat.start, ending.start)

                else:
                    # add the first occurrence of the repeat
                    sv.add_segment(repeat.start, repeat.end)

                    # no: add the full interval of the repeat (the second time)
                    new_sv.add_segment(repeat.start, repeat.end)
                    new_sv.add_segment(repeat.start, repeat.end)

                # this repeat has been handled, update the score time
                t_score = repeat.end

                # add both score variants
                new_svs.append(sv)
                new_svs.append(new_sv)

            svs = new_svs

        # are we at the end of the piece already?
        if t_score < end_point:
            # no, append the interval from the current score
            # position to the end of the piece
            for sv in svs:
                sv.add_segment(t_score, end_point)

        return svs

    def test_timeline(self):
        """
        Test if all ending objects have occurred as starting object as
        well.
        """

        s = set()
        for tp in self.timeline.points:
            for k, oo in tp.starting_objects.items():
                for o in oo:
                    s.add(o)
            for k, oo in tp.ending_objects.items():
                for o in oo:
                    assert o in s
                    s.remove(o)
        LOGGER.info('Timeline is OK')

    def _make_repeat_structure(self):
        """
        Return a list of sequence times based on the repeat structure
        of the piece, that can be used to create an unfolded timeline.

        Returns
        -------
        list
            A list of triples (s, e, o), where s is the score start
            time of a segment, e is the score end time of a segment,
            and o is the absolute (score variant) start time of that
            segment in the unfolded score

        """
        LOGGER.warning('Generation of repeat structures involving da'
                       ' capo/fine/coda/segno directions is not (properly)'
                       ' implemented yet')

        repeats = self.timeline.get_all_of_type(Repeat)
        dacapos = self.timeline.get_all_of_type(DaCapo)
        fines = self.timeline.get_all_of_type(Fine)

        if len(dacapos) > 0:
            dacapo = dacapos[0]
        else:
            dacapo = None

        if len(fines) > 0:
            fine = fines[0]
        else:
            fine = None

        # t_score is used to keep the time in the score
        t_score = TimePoint(0)
        # the last time instance in the piece
        end_point = self.timeline.points[-1]
        # t_unfold is used to keep the time in the score variant
        t_unfold = 0
        # times will aggregate the triples that make up the result
        times = []
        # flag that tells... if we've reached a "da capo" sign in the
        # score
        reached_dacapo = False

        # each repeat holds start and end time of a score interval to
        # be repeated
        for repeat in repeats:

            # is the start of the repeat after our current score
            # position?
            if repeat.start > t_score:
                # yes: add the tuple (t_score, repeat.start) to the
                # result this is the span before the interval that is
                # to be repeated
                times.append((t_score, repeat.start, t_unfold))
                # increase t_unfold by the interval [t_score,
                # repeat.start]
                t_unfold += (repeat.start.t - t_score.t)

            # add the first occurrence of the repeat
            times.append((repeat.start, repeat.end, t_unfold))
            # update t_unfold accordingly
            t_unfold += (repeat.end.t - repeat.start.t)

            # is there a da capo within the repeat interval?
            if dacapo is not None and repeat.start < dacapo.start <= repeat.end:
                # yes: set the reached_dacapo flag
                reached_dacapo = True
                # play the second time only up to the da capo, and
                # stop processing further repeats
                times.append((repeat.start, dacapo.start, t_unfold))
                # update t_unfold accordingly
                t_unfold += (dacapo.start.t - repeat.start.t)

                break

            # get any "endings" (e.g. 1 / 2 volta) of the repeat
            # (there are not supposed to be more than one)
            endings = repeat.end.get_ending_objects_of_type(Ending)

            # is there an ending?
            if len(endings) > 0:
                # yes
                ending = endings[0]
                # we are in the second iteration of the repeat, so
                # only add the interval of the repeat up to the ending
                # (rather than up to the end of the repeat)
                times.append((repeat.start, ending.start, t_unfold))
                # update t_unfold accordingly
                t_unfold += (ending.start.t - repeat.start.t)
            else:
                # no: add the full interval of the repeat (the second time)
                times.append((repeat.start, repeat.end, t_unfold))
                # update t_unfold accordingly
                t_unfold += (repeat.end.t - repeat.start.t)

            # this repeat has been handled, update the score time
            t_score = repeat.end

        # are we at a da capo sign?
        if reached_dacapo:
            # yes; is there a fine?
            if fine is not None:
                # yes

                # get the notes starting at the fine sign
                notes = fine.start.get_starting_objects_of_type(Note)

                # TODO: the following appears to be incorrect, the
                # musicxml spec says the fine *follows* the last notes
                # to be played, so the end point should always be the
                # time instance of the fine sign, unless otherwise stated:

                # TODO: if "fine" is a number, treat it as the quarter
                # duration that all final notes are supposed to have,
                # rather than have all the notes keep their own
                # duration

                # are there any notes starting at the fine sign?
                if len(notes) > 0:
                    # yes: get the off times
                    off_times = np.array([n.end.t for n in notes])
                    # set the end point of the next interval to the
                    # latest off time
                    end_point = notes[np.argmax(off_times)].end
                else:
                    # no: set the end point of the next interval to
                    # the time of the fine sign
                    end_point = fine.start

            # add the interval from the start of the piece to
            # end_point, which is either:
            # 1. the end of the piece (no fine sign)
            # 2. the time of the fine sign (no notes start at fine sign)
            # 3. the offset of the longest note played at a fine sign (notes
            # start at fine sign)
            times.append((self.timeline.points[0], end_point, t_unfold))
        else:
            # not at a da capo sign

            # are we at the end of the piece already?
            if t_score < end_point:
                # no, append the interval from the current score
                # position to the end of the piece
                times.append((t_score, end_point, t_unfold))

        # for s, e, o in times:
        #     print(s.t, e.t, o)

        return times

    def unfold_timeline(self):
        """
        Return a new TimeLine, where all repeat structures are
        unfolded. This includes 1/2 endings (volta brackets),
        and Da Capo al Fine structures. In this new timeline, both the
        timepoints and the musical objects are copied to unfold the
        structure. Note that the ID attributes of the musical objects
        are copied along, so these ID's will not be unique (but the
        duplicate ID's may be useful to identify which objects are
        duplicates of which).

        Returns
        -------
        tl : TimeLine object
            A TimeLine object containing the unfolded timepoints
        """

        self.test_timeline()

        new_timeline = []
        ending_objects_tmp = defaultdict(list)

        def add_points_between(start, end, offset, prev_ending_objects,
                               object_map, include_end=False):
            # print('add_points_between',start.t, end.t, offset, include_end)

            end_operator = operator.le if include_end else operator.lt

            point_idx = np.logical_and(
                operator.ge(self.timeline.points, start),
                end_operator(self.timeline.points, end))

            # make a copy of all timepoints in the selected range
            new_points = np.array([copy(x)
                                   for x in self.timeline.points[point_idx]])

            for i, tp in enumerate(new_points):
                # let the range start at offset
                tp.t = tp.t - start.t + offset

                # make a copy of all starting objects, for the new
                # objects, set the start attribute to the new
                # timepoint, and set the new objects to be the
                # starting objects of the new timepoint
                new_starting = defaultdict(list)
                for k, objects in tp.starting_objects.items():
                    new_objects = [copy(o) for o in objects]

                    for o in new_objects:
                        o.start = tp

                    object_map.update(zip(objects, new_objects))
                    new_starting[k] = new_objects

                tp.starting_objects = new_starting

                if i > 0:
                    new_ending = defaultdict(list)
                    for k, objects in tp.ending_objects.items():
                        new_objects = [object_map[o]
                                       for o in objects]

                        for o in new_objects:
                            o.end = tp

                        new_ending[k] = new_objects
                    tp.ending_objects = new_ending

            if len(new_points) > 0:
                # print('setting ending objects from last repeat:')
                # print(new_points[0].t)
                new_points[0].ending_objects = prev_ending_objects
                for k, oo in prev_ending_objects.items():
                    for o in oo:
                        o.end = new_points[0]

            ending_objects_copy = defaultdict(list)
            for k, oo in end.ending_objects.items():
                ending_objects_copy[k] = [object_map[o] for o in oo]
            return new_points, ending_objects_copy, object_map

        o_map = {}

        segments = self._make_repeat_structure()
        N = len(segments)

        for i, (start, end, offset) in enumerate(segments):

            include_end = i == N - 1

            new_points, ending_objects_tmp, o_map = \
                add_points_between(
                    start, end, offset, ending_objects_tmp, o_map, include_end)

            new_timeline.append(new_points)

            # for new_points in new_timeline:
            #     for i,p in enumerate(new_points):
            #         for n in p.get_starting_objects_of_type(Note):
            #             if n.duration > 130:
            #                 print(i, len(new_points))
            #                 print(n)
            #                 print('',n)
            #                 assert 1 == 0
        new_timeline = np.concatenate(new_timeline)

        for i in range(1, len(new_timeline)):
            new_timeline[i - 1].next = new_timeline[i]
            new_timeline[i].prev = new_timeline[i - 1]

        new_timeline[0].prev = None
        new_timeline[-1].next = None

        # assert np.all(np.diff(np.array([tp.t for tp in new_timeline])) > 0)

        tl = TimeLine()
        tl.points = new_timeline

        # for tp in tl.points:
        #     print(tp)
        #     for n in tp.get_starting_objects_of_type(Note):
        #         print(n.start.t, tp.t, n.end.t)
        #         assert n.start.t <= n.end.t

        return tl

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
            the type of grace note, if no type is specified. Possibilites
            are: {'appoggiatura', 'acciaccatura'}.

        min_steal : float, optional
            the min steal proportion if no proportion is specified

        max_steal : float, optional
            the max steal proportion if no proportion is specified
        """

        assert default_type in (u'appoggiatura', u'acciaccatura')
        assert 0 < min_steal <= max_steal
        assert min_steal <= max_steal < 1.0

        def n_notes_to_steal(n_notes):
            return min_steal + (max_steal - min_steal) * 2 * (1 / (1 + np.exp(- n_notes + 1)) - .5)

        # def shorten_main_notes_by(dur_prop, notes, group_id):
        #     # start and duration of the main note
        #     old_start = notes[0].start
        #     n_dur = np.min([n.duration for n in notes])
        #     new_start_t = old_start.t + n_dur * dur_prop
        #     print(n_dur * dur_prop)
        #     for i, n in enumerate(notes):
        #         old_start.starting_objects[Note].remove(n)
        #         self.timeline.add_starting_object(new_start_t, n)
        #         n.appoggiatura_group_id = group_id
        #         n.appoggiatura_duration = dur_prop
        #     return new_start_t

        def shorten_main_notes_by(offset, notes, group_id):
            # start and duration of the main note
            old_start = notes[0].start
            n_dur = np.min([n.duration for n in notes])
            # print('app', n_dur, offset)
            offset = min(n_dur * .5, offset)
            new_start_t = old_start.t + offset
            for i, n in enumerate(notes):
                old_start.starting_objects[Note].remove(n)
                self.timeline.add_starting_object(new_start_t, n)
                n.appoggiatura_group_id = group_id
                n.appoggiatura_duration = offset / float(n_dur)
            return new_start_t

        # def shorten_prev_notes_by(dur_prop, notes, group_id):
        #     old_end = notes[0].end
        #     n_dur = notes[0].duration
        #     new_end_t = old_end.t - n_dur * dur_prop

        #     for n in notes:
        #         old_end.ending_objects[Note].remove(n)
        #         self.timeline.add_ending_object(new_end_t, n)
        #         n.acciaccatura_group_id = group_id
        #         n.acciaccatura_duration = dur_prop

        #     return new_end_t

        def shorten_prev_notes_by(offset, notes, group_id):
            old_end = notes[0].end
            n_dur = notes[0].duration
            #print('acc', n_dur, offset)
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
                n.start.starting_objects[Note].remove(n)
                self.timeline.add_starting_object(times[i], n)
                n.end.ending_objects[Note].remove(n)
                self.timeline.add_ending_object(times[i + 1], n)
                n.appoggiatura_group_id = group_id
                n.appoggiatura_idx = i
                n.appoggiatura_size = N

        self.timeline.unlock()

        grace_notes = [n for n in self.notes if n.grace_type is not None]
        time_grouped_gns = partition(
            operator.attrgetter('start.t'), grace_notes)
        times = sorted(time_grouped_gns.keys())

        group_counter = 0
        for t in times:

            voice_grouped_gns = partition(operator.attrgetter('voice'),
                                          time_grouped_gns[t])
            # print(t)
            for voice, gn_group in voice_grouped_gns.items():
                # print('  voice {}'.format(voice))
                for n in gn_group:
                    if n.grace_type == 'grace':
                        n.grace_type = default_type

                type_grouped_gns = partition(operator.attrgetter('grace_type'),
                                             gn_group)

                for gtype, type_group in type_grouped_gns.items():
                    total_steal_old = n_notes_to_steal(len(type_group))
                    total_steal = np.sum([n.duration_from_symbolic for n
                                          in type_group])
                    # print("n_notes, old, new", len(type_group), total_steal_old, total_steal)

                    # print('    {}: {} {:.3f}'.format(gtype, len(type_group),
                    # total_steal))
                    main_notes = [m for m in type_group[0].simultaneous_notes_in_voice
                                  if m.grace_type is None]
                    # multip
                    if len(main_notes) > 0:
                        # total_steal =
                        total_steal = min(
                            main_notes[0].duration / 2., total_steal)

                    if gtype == 'appoggiatura':
                        # main_notes = [m for m in type_group[0].simultaneous_notes_in_voice
                        #               if m.grace_type is None]
                        # print(total_steal, len(type_group))
                        total_steal = np.sum([n.duration_from_symbolic for n
                                              in type_group])
                        # if len(main_notes) == 0:
                        #     main_notes = [m for m in type_group[0].next_notes_in_voice
                        #                   if m.grace_type is None]
                        # print('    main: {}'.format(len(main_notes)))
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

        self.timeline.link()
        self.timeline.lock()

    def pprint(self, l=0):
        pre = u'  ' * l
        s = [u'{}{} ({})'.format(pre, self.part_name, self.part_id)]
        bm = self.beat_map

        for tp in self.timeline.points:
            #s.append(pre + tp.__unicode__() + u'(beat: {0})'.format(bm(tp.t)))
            # s.append(u'{}{}(beat: {})'.format(pre, tp, bm(tp.t)[0]))
            s.append('{}{}(beat: {})'.format(pre, tp, bm(tp.t)))
            for cls, objects in list(tp.starting_objects.items()):
                if len(objects) > 0:
                    #s.append(pre + u'  {0}'.format(cls.__name__))
                    s.append('{}  {}'.format(pre, cls.__name__))
                    for o in objects:
                        #s.append(pre + u'    {0}'.format(o))
                        s.append('{}    {}'.format(pre, o))
            s.append(u' Stop')
            for cls, objects in list(tp.ending_objects.items()):
                if len(objects) > 0:
                    #s.append(pre + u'  {0}'.format(cls.__name__))
                    s.append(u'{}  {}'.format(pre, cls.__name__))
                    for o in objects:
                        #s.append(pre + u'    {0}'.format(o))
                        s.append(u'{}    {}'.format(pre, o))
        return u'\n'.join(s)

    def _get_beat_map(self, quarter=False, default_div=1, default_den=4):
        """
        This returns an interpolator that will accept as input timestamps
        in divisions and returns these timestamps' beatnumbers. If the flag
        `quarter` is used, these beatnumbers will refer to quarter note steps.

        Parameters
        ----------
        quarter : boolean, optional. Default: False

        Returns
        -------
        scipy interpolate interp1d object

       """

        if len(self.timeline.points) == 0:
            return None

        try:
            first_measure = self.timeline.points[
                0].get_starting_objects_of_type(Measure)[0]
            if first_measure.upbeat:
                offset = -first_measure.get_measure_duration(quarter=quarter)
            else:
                offset = 0
        except IndexError:
            offset = 0

        divs = np.array(
            [(x.start.t, x.divs) for x in
             self.timeline.get_all_of_type(Divisions)], dtype=np.int)

        dens = np.array(
            [(x.start.t, np.log2(x.beat_type)) for x in
             self.timeline.get_all_of_type(TimeSignature)], dtype=np.int)

        if divs.shape[0] == 0:
            LOGGER.warning(("No Divisions found in ScorePart, "
                            "assuming divisions = {0}").format(default_div))
            divs = np.array(((0, default_div),), dtype=np.int)

        if dens.shape[0] == 0:
            LOGGER.warning(("No TimeSignature found in ScorePart, "
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
            dens[:, 1] = 1

        # integrate second column, where first column is time:
        # new_divs = np.cumsum(np.diff(divs[:, 0]) * divs[:-1, 1])
        new_divs = divide_outside_cumsum(divs)

        divs = divs.astype(np.float)
        divs[1:, 1] = new_divs
        divs[0, 1] = divs[0, 0]

        # at this point divs[:, 0] is a list of musicxml div times
        # and divs[:, 1] is a list of corresponding quarter note times

        # interpolation object to map div times to quarter times:
        # div_intp = my_interp1d(divs[:, 0], divs[:, 1])
        div_intp = interp1d(divs[:, 0], divs[:, 1])

        dens = dens.astype(np.float)
        # change dens[:, 0] from div to quarter times
        dens[:, 0] = div_intp(dens[:, 0])
        # change dens[:, 1] back from log2(beat_type) to beat_type and divide by
        # 4; Here take the reciprocal (4 / 2**dens[:, 1]) since in divid_outside_cumsum we will be
        # dividing rather than multiplying:
        dens[:, 1] = 4 / 2**dens[:, 1]

        # dens_new = np.cumsum(np.diff(dens[:, 0]) * dens[:-1, 1])
        dens_new = divide_outside_cumsum(dens)
        dens[1:, 1] = dens_new
        dens[0, 1] = dens[0, 0]

        den_intp = interp1d(dens[:, 0], dens[:, 1])

        if len(self.timeline.points) < 2:
            return lambda x: np.zeros(len(x))
        else:
            def f(x):
                try:
                    # divi = div_intp(x)
                    # deni = den_intp(divi) + offset
                    # np.savetxt('/tmp/bm.txt', np.column_stack((x, divi, deni)), fmt="%.3f")
                    # np.savetxt('/tmp/den.txt', dens, fmt="%.3f")
                    # return deni
                    return den_intp(div_intp(x)) + offset

                except ValueError:
                    print(np.min(x), np.max(x))
                    raise
            return f

    def _get_notes(self, unfolded=False):
        """
        return all note objects of the score part.

        Parameters
        ----------
        unfolded : boolean, optional. Default: False
            whether to unfolded the timeline or not.

        Returns
        -------
        notes : list of Note objects

        """
        notes = []
        if unfolded:
            tl = self.unfold_timeline()
        else:
            tl = self.timeline
        for tp in tl.points:
            notes.extend(tp.get_starting_objects_of_type(Note) or [])
        return notes

    def get_loudness_directions(self):
        """
        return all loudness directions

        """
        return self.timeline.get_all_of_type(LoudnessDirection, include_subclasses=True)
        # directions = []
        # for tp in self.timeline.points:
        #     directions.extend(
        #         tp.get_starting_objects_of_type(DynamicLoudnessDirection) or [])
        #     directions.extend(
        #         tp.get_starting_objects_of_type(ConstantLoudnessDirection) or [])
        #     directions.extend(
        #         tp.get_starting_objects_of_type(ImpulsiveLoudnessDirection) or [])
        # return directions

    def get_tempo_directions(self):
        """
        return all tempo directions

        """
        return self.timeline.get_all_of_type(TempoDirection, include_subclasses=True)
        # directions = []
        # for tp in self.timeline.points:
        #     directions.extend(
        #         tp.get_starting_objects_of_type(DynamicTempoDirection) or [])
        #     directions.extend(
        #         tp.get_starting_objects_of_type(ConstantTempoDirection) or [])
        # return directions

    # @property
    @cached_property
    def notes(self):
        """
        all note objects
        """
        return self._get_notes()

    @cached_property
    def notes_unfolded(self):
        """
        all note objects, with unfolded timeline.
        """
        return self._get_notes(unfolded=True)

    # @cached_property
    @property
    def beat_map(self):
        """
        map timeline times to beat times
        """
        return self._get_beat_map()

    # @cached_property
    @property
    def quarter_map(self):
        """
        map timeline times to beat times
        """
        return self._get_beat_map(quarter=True)
