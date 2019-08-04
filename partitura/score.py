#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This module contains an ontology to represent musical scores. A score
is defined at the highest level by a Part object. This object
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
from collections import defaultdict
import logging
import operator
import itertools
from numbers import Number

import numpy as np
from scipy.interpolate import interp1d

from partitura.utils import ComparableMixin, partition, iter_subclasses

# the score ontology for longer scores requires a high recursion limit
# increase when needed
sys.setrecursionlimit(100000)

LOGGER = logging.getLogger(__name__)
# _NON_ALPHA_NUM_PAT = re.compile(r'\W', re.UNICODE)
_LABEL_DURS = {
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
_DOT_MULTIPLIERS = (1, 1+1/2., 1+3/4., 1+7/8.)
_MIDI_BASE_CLASS = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}
_MORPHETIC_BASE_CLASS = {'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6}
_MORPHETIC_OCTAVE = {0: 32, 1: 39, 2: 46, 3: 53, 4: 60, 5: 67, 6: 74, 7: 81, 8: 89}
_ALTER_SIGNS = {None: '', 1: '#', 2: 'x', -1: 'b', -2: 'bb'}

def format_symbolic_duration(symbolic_dur):
    # TODO: can symbolic_dur be None?
    return symbolic_dur['type']+'.'*symbolic_dur['dots']


def symbolic_to_numeric_duration(symbolic_dur, divs):
    # TODO: can symbolic_dur be None?
    numdur = divs * _LABEL_DURS[symbolic_dur.get('type', None)]
    numdur *= _DOT_MULTIPLIERS[symbolic_dur.get('dots', 0)]
    numdur *= (symbolic_dur.get('normal_notes', 1) / 
               symbolic_dur.get('actual_notes', 1))
    return numdur


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
        # if not (i < len(self.points) and self.points[i].t == tp.t):
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
        
    def get_all(self, cls, start=None, end=None, include_subclasses=False):
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
                end = TimePoint(end)
            end_idx = np.searchsorted(self.points, end, side='left')
        else:
            end_idx = len(self.points)

        r = []
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
    functionality to add, and remove TimePoints.

    """

    def __init__(self, t, label=''):
        self.t = t
        self.label = label
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
        super().__init__()
        self.nr = nr

    def __str__(self):
        return 'page {0}'.format(self.nr)


class System(TimedObject):

    def __init__(self, nr=0):
        super().__init__()
        self.nr = nr

    def __str__(self):
        return 'system {0}'.format(self.nr)

class Clef(TimedObject):

    def __init__(self, number, sign, line, octave_change):
        super().__init__()
        self.number = number
        self.sign = sign
        self.line = line
        self.octave_change = octave_change

    def __str__(self):
        return 'clef {} on line {} (number {})'.format(self.sign, self.line, self.number)
    

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

    def __str__(self):
        return 'slur at voice {0} (ends at {1})'.format(self.voice, self.end and self.end.t)


class Repeat(TimedObject):

    def __init__(self):
        super(Repeat, self).__init__()

    def __str__(self):
        return 'Repeat (from {0} to {1})'.format(self.start and self.start.t, self.end and self.end.t)


class DaCapo(TimedObject):

    def __init__(self):
        super(DaCapo, self).__init__()

    def __str__(self):
        return u'Dacapo'


class Fine(TimedObject):

    def __init__(self):
        super(Fine, self).__init__()

    def __str__(self):
        return 'Fine'


class Fermata(TimedObject):

    def __init__(self):
        super(Fermata, self).__init__()

    def __str__(self):
        return 'Fermata'


class Ending(TimedObject):

    """
    Object that represents one part of a 1---2--- type ending of a
    musical passage (aka Volta brackets).
    """

    def __init__(self, number):
        super(Ending, self).__init__()
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
        super(Measure, self).__init__()
        self.number = number


    def __str__(self):
        return 'measure {0} at page {1}, system {2}'.format(self.number, self.page, self.system)

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
        Parameters
        ----------
        quarter : ????, optional. Default: False

        Returns
        -------

        """

        if self.start.next is None:
            LOGGER.warning('Measure start has no successor')
            return 0
        
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
        nextm = self.start.get_next_of_type(Measure)

        invalid = False
        if len(divs) == 0:
            LOGGER.warning('Part specifies no divisions')
            invalid = True
        if len(ts) == 0:
            LOGGER.warning('Part specifies no time signatures')
            invalid = True
        if len(nextm) == 0:
            LOGGER.warning('Part has just one measure')
            invalid = True

        if invalid:
            LOGGER.warning(
                'could not be determine if meaure is incomplete, assuming complete')
            return False

        measure_dur = nextm[0].start.t - self.start.t
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
        super(TimeSignature, self).__init__()
        self.beats = beats
        self.beat_type = beat_type

    def __str__(self):
        return 'time signature: {0}/{1}'.format(self.beats, self.beat_type)


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

    def __str__(self):
        return 'divisions: quarter={0}'.format(self.divs)


class Tempo(TimedObject):

    def __init__(self, bpm):
        super(Tempo, self).__init__()
        self.bpm = bpm

    def __str__(self):
        return 'tempo: bpm={0}'.format(self.bpm)


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
        super(KeySignature, self).__init__()
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
        return f'key signature: fifths={self.fifths}, mode={self.mode} ({self.name})'


    
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

    def __str__(self):
        return 'transposition: diatonic={0}, chromatic={1}'.format(self.diatonic, self.chromatic)


class Words(TimedObject):

    """
    Parameters
    ----------
    text : str
    """

    def __init__(self, text):
        super(Words, self).__init__()
        self.text = text

    def __str__(self):
        return '{}: {}'.format(type(self).__name__, self.text)


class Direction(TimedObject):

    """

    """

    def __init__(self, text):
        self.text = text
        self.start = None
        self.end = None

    def __str__(self):
        return '{}: {}'.format(type(self).__name__, self.text)


class TempoDirection(Direction): pass


class DynamicTempoDirection(TempoDirection):
    def __init__(self, text):
        Direction.__init__(self, text)
        self.intermediate = []


class ConstantTempoDirection(TempoDirection): pass


class ResetTempoDirection(ConstantTempoDirection): pass


class LoudnessDirection(Direction): pass


class DynamicLoudnessDirection(LoudnessDirection):
    def __init__(self, text):
        Direction.__init__(self, text)
        self.intermediate = []

class ConstantLoudnessDirection(LoudnessDirection): pass


class ImpulsiveLoudnessDirection(LoudnessDirection): pass


class Chord(TimedObject):
    """
    Represents a chord
    """
    def __init__(self):
        # TODO: add references to constituent notes
        pass


class GenericNote(TimedObject):
    """
    Represents the common aspects of notes and rests (and in the future unpitched notes)

    Parameters
    ----------
    voice : integer, optional (default: None)

    id : integer, optional (default: None)

    """
    def __init__(self, id=None, voice=None, staff=None, symbolic_duration=None):
        self.voice = voice
        self.id = id
        self.staff = staff
        self.symbolic_duration = symbolic_duration
        self.tie_prev = None
        self.tie_next = None
    
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
        s = f'{self.id} voice={self.voice} staff={self.staff} type="{format_symbolic_duration(self.symbolic_duration)}"'
        if self.tie_prev or self.tie_next:
            all_tied = self.tie_prev_notes + [self] + self.tie_next_notes
            tied_dur = '+'.join(format_symbolic_duration(n.symbolic_duration) for n in all_tied)
            tied_id = '+'.join(n.id or 'None' for n in all_tied)
            return s + f' tied: {tied_id}'
        else:
            return s
    #     return ('{0}{1}{2} ({8}-{9}, midi: {3}, duration: {5} ({10}), voice: {4}, id: {6}, {7}) {11}'
    #             .format(self.step, self.alter_sign, self.octave,
    #                     self.midi_pitch, self.voice, self.duration,
    #                     self.id or '', self.grace_type if self.grace_type else '',
    #                     self.start and self.start.t, self.end and self.end.t,
    #                     tied_dur, tied_id))

class Note(GenericNote):
    def __init__(self, step, alter, octave, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = step
        self.alter = alter
        self.octave = octave

    def __str__(self):
        return ' '.join((super().__str__(),
                         f'pitch={self.step}{self.alter_sign}{self.octave}'))
    #     return ('{0}{1}{2} ({8}-{9}, midi: {3}, duration: {5} ({10}), voice: {4}, id: {6}, {7}) {11}'
    #             .format(self.step, self.alter_sign, self.octave,
    #                     self.midi_pitch, self.voice, self.duration,

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


class Rest(GenericNote):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class GraceNote(Note):
    def __init__(self, grace_type, *args, steal_proportion=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grace_type = grace_type
        self.steal_proportion = steal_proportion

# class Note(TimedObject):

#     """
#     Represents a note

#     Parameters
#     ----------
#     step : str
#         the basic pitch class, like 'C', 'D', 'E', etc.

#     alter: integer
#         number of semi-tones to alterate the note from its basic pitch
#         given by `step`.
#         Note that the musicxml standard in principle allows for this to
#         be a float number for microtones (micro-intonation). In Midi this
#         would/could then translate to a pitch-bend.

#     octave : integer
#         the octave where octave 4 is the one having middle C (C4).

#     voice : integer, optional. Default: None

#     id : integer, optional. Default: None

#     ...


#     Attributes
#     ----------
#     previous_notes_in_voice :

#     simultaneous_notes_in_voice :

#     next_notes_in_voice :

#     midi_pitch : integer

#     morphetic_pitch :

#     alter_sign :

#     duration :

#     """

#     def __init__(self, step, alter, octave, voice=None, id=None,
#                  symbolic_duration=None,
#                  grace_type=None, steal_proportion=None,
#                  staccato=False, fermata=False, accent=False,
#                  staff=None):
#         self.step = step
#         # alter_values = (None, 0, 1, 2, 3, -1, -2, 3)
#         # if alter not in alter_values:
#         #     raise Exception('alter should be one of {}'.format(alter_values))
#         self.alter = alter
#         if alter == 0:
#             alter = None
#         self.octave = octave
#         self.voice = voice
#         self.id = id
#         self.grace_type = grace_type
#         self.steal_proportion = steal_proportion
#         self.staccato = staccato
#         self.fermata = fermata
#         self.accent = accent
#         self.staff = staff
#         # self.symbolic_durations = []
#         # if symbolic_duration is not None:
#         #     self.symbolic_durations.append(symbolic_duration)
#         self.symbolic_duration = symbolic_duration
#         self.tie_prev = None
#         self.tie_next = None

#     @property
#     def previous_notes_in_voice(self):
#         n = self
#         while True:
#             nn = n.start.get_prev_of_type(Note)
#             if nn == []:
#                 return nn
#             else:
#                 voice_notes = [m for m in nn if m.voice == self.voice]
#                 if len(voice_notes) > 0:
#                     return voice_notes
#                 n = nn[0]

#     @property
#     def simultaneous_notes_in_voice(self):
#         return [m for m in self.start.starting_objects[Note]
#                 if m.voice == self.voice and m != self]

#     @property
#     def next_notes_in_voice(self):
#         n = self
#         while True:
#             nn = n.start.get_next_of_type(Note)
#             if nn == []:
#                 return nn
#             else:
#                 voice_notes = [m for m in nn if m.voice == self.voice]
#                 if len(voice_notes) > 0:
#                     return voice_notes
#                 n = nn[0]

#     @property
#     def midi_pitch(self):
#         """
#         the midi pitch value of the note (MIDI note number).
#         C4 (middle C, in german: c') is note number 60.

#         Returns
#         -------
#         integer
#             the note's pitch as MIDI note number.
#         """
#         base_class = {'c': 0, 'd': 2, 'e': 4, 'f': 5,
#                       'g': 7, 'a': 9, 'b': 11}[self.step.lower()] + (self.alter or 0)

#         return (self.octave + 1) * 12 + base_class

#     @property
#     def morphetic_pitch(self):
#         """
#         the morphetic value of the note, i.e. a single integer.
#         It corresponds to the (vertical) position of the note in
#         the barline system.

#         Returns
#         -------
#         integer
#         """
#         base_class = {'c': 0, 'd': 1, 'e': 2, 'f': 3,
#                       'g': 4, 'a': 5, 'b': 6}[self.step.lower()]
#         octave_number = {0: 32, 1: 39, 2: 46, 3: 53,
#                          4: 60, 5: 67, 6: 74, 7: 81,
#                          8: 89}[self.octave]

#         return octave_number + base_class

#     @property
#     def alter_sign(self):
#         """
#         the alteration of the note

#         Returns
#         -------
#         str
#         """
#         return {None: '', 1: '#', 2: 'x', -1: 'b', -2: 'bb'}[self.alter]

#     @property
#     def duration(self):
#         """
#         the duration of the note in divisions

#         Returns
#         -------
#         number
#         """

#         try:
#             return self.end.t - self.start.t
#         except:
#             LOGGER.warn('no end time found for note')
#             return 0

#     @property
#     def duration_from_symbolic(self):
#         divs = self.start.get_prev_of_type(Divisions, True)
#         if len(divs) == 0:
#             div = 1
#         else:
#             div = divs[0].divs
#         return symbolic_to_numeric_duration(self.symbolic_duration, div)
#         # TODO: it is theoretically possible that the divisions change
#         # in between tied notes. The current code assumes this does not happen.
#         # return symbolic_to_numeric_duration(self.symbolic_durations, div)
#         # return sum(symbolic_to_numeric_duration(sd, div) for sd in self.symbolic_durations)

    # def __str__(self):
    #     all_tied = self.tie_prev_notes + [self] + self.tie_next_notes
    #     tied_dur = '+'.join(format_symbolic_duration(n.symbolic_duration) for n in all_tied)
    #     tied_id = '+'.join(n.id or 'None' for n in all_tied)
    #     return ('{0}{1}{2} ({8}-{9}, midi: {3}, duration: {5} ({10}), voice: {4}, id: {6}, {7}) {11}'
    #             .format(self.step, self.alter_sign, self.octave,
    #                     self.midi_pitch, self.voice, self.duration,
    #                     self.id or '', self.grace_type if self.grace_type else '',
    #                     self.start and self.start.t, self.end and self.end.t,
    #                     tied_dur, tied_id))


# class Rest(Note):
#     """
#     Represents a rest. It is defined as a subclass of Note so that it can be
#     conveniently retrieved from a timeline along with Note objects if needed.
#     """
#     def __init__(self, voice=None, id=None,
#                  symbolic_duration=None,
#                  staff=None):
#         self.voice = voice
#         self.id = id
#         self.staff = staff
#         self.symbolic_duration = symbolic_duration

#     def __str__(self):
#         return ('rest (id: {}, duration: {}; voice: {}; staff: {})'
#                 .format(self.id, format_symbolic_duration(self.symbolic_duration),
#                         self.voice, self.staff))


def get_all_parts(constituents):
    """
    From a list whose elements are either Part objects or
    PartGroup objects, return an ordered list of Part objects.

    Parameters:
    -----------
    constituents : iterable
        a list of Part/PartGroup objects

    Returns:
    --------
    iterable
        a list of all Part objects embedded in `constituents`

    """

    return [part for constituent in constituents
            for part in
            ((constituent,) if isinstance(constituent, Part)
             else get_all_parts(constituent.constituents))]


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

    parts : list of Part objects
        a list of all Part objects in this PartGroup
    """

    def __init__(self, grouping_symbol=None, name=None):
        self.grouping_symbol = grouping_symbol
        self.constituents = []
        self.name = name
        self.parent = None
        self.number = None

    @property
    def parts(self):
        return get_all_parts(self.constituents)

    def pprint(self, l=0):
        if self.name is not None:
            name_str = ' / {0}'.format(self.name)
        else:
            name_str = ''
        s = ['    ' * l + '{0}{1}'.format(self.grouping_symbol, name_str)]
        for ch in self.constituents:
            s.append(ch.pprint(l + 1))
        return '\n'.join(s)


class ScoreVariant(object):

    def __init__(self, start_time=0):
        self.t_unfold = start_time
        self.segments = []

    def add_segment(self, start, end):
        self.segments.append((start, end, self.t_unfold))
        self.t_unfold += (end.t - start.t)

    def get_segments(self):
        """
        Return segment (start, end, offset) information for each of the segments in
        the score variant.
        """
        return [(s.t, e.t, 0 if i > 0 else o)
                for i, (s, e, o) in enumerate(self.segments)]

    def clone(self):
        """
        Return a clone of the ScoreVariant
        """
        clone = ScoreVariant(self.t_unfold)
        clone.segments = self.segments[:]
        return clone


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

        repeats = self.timeline.get_all(Repeat)

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

        repeats = self.timeline.get_all(Repeat)
        dacapos = self.timeline.get_all(DaCapo)
        fines = self.timeline.get_all(Fine)

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

        new_timeline = np.concatenate(new_timeline)

        for i in range(1, len(new_timeline)):
            new_timeline[i - 1].next = new_timeline[i]
            new_timeline[i].prev = new_timeline[i - 1]

        new_timeline[0].prev = None
        new_timeline[-1].next = None

        # assert np.all(np.diff(np.array([tp.t for tp in new_timeline])) > 0)

        tl = TimeLine()
        tl.points = new_timeline

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
                n.start.starting_objects[Note].remove(n)
                self.timeline.add_starting_object(times[i], n)
                n.end.ending_objects[Note].remove(n)
                self.timeline.add_ending_object(times[i + 1], n)
                n.appoggiatura_group_id = group_id
                n.appoggiatura_idx = i
                n.appoggiatura_size = N

        grace_notes = [n for n in self.notes if n.grace_type is not None]
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
                                  if m.grace_type is None]

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


    def pprint(self, l=1):
        pre = '    ' * l
        s = ['{} ({})'.format(self.part_name, self.part_id)]
        bm = self.beat_map

        for tp in self.timeline.points:
            s.append('\n{}{}(beat: {})'.format(pre, tp, bm(tp.t)))
            s.append('{}Start'.format(pre))
            for cls, objects in list(tp.starting_objects.items()):
                if len(objects) > 0:
                    s.append('{}  {}'.format(pre, cls.__name__))
                    for o in objects:
                        s.append('{}    {}'.format(pre, o))
            s.append('{}Stop'.format(pre))
            for cls, objects in list(tp.ending_objects.items()):
                if len(objects) > 0:
                    s.append(u'{}  {}'.format(pre, cls.__name__))
                    for o in objects:
                        s.append(u'{}    {}'.format(pre, o))
        return u'\n'.join(s)


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
            tl = self.unfold_timeline()
        else:
            tl = self.timeline
        return tl.get_all(cls, include_subclasses=include_subclasses)

    @property
    def notes(self):
        """
        Return all note objects of the score part.

        Returns
        -------
        list
            list of Note objects

        """
        return self.list_all(Note, unfolded=False)

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
