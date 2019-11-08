#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This module defines an ontology of musical elements to represent
musical scores, such as measures, notes, slurs, words, tempo and
loudness directions. A score is defined at the highest level by a
`Part` object (or a hierarchy of `Part` objects, in a `PartGroup`
object). This object serves as a timeline at which musical elements
are registered in terms of their start and end times.

"""

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
    ReplaceRefMixin,
    partition,
    iter_subclasses,
    iter_current_next,
    sorted_dict_items,
    PrettyPrintTree,
    MIDI_BASE_CLASS,
    ALTER_SIGNS,
    find_tie_split,
    format_symbolic_duration,
    estimate_symbolic_duration,
    symbolic_to_numeric_duration,
    fifths_mode_to_key_name,
    pitch_spelling_to_midi_pitch,
    to_quarter_tempo
)

LOGGER = logging.getLogger(__name__)


class Part(object):
    """Represents a score part, e.g. all notes of one single instrument
    (or multiple instruments written in the same staff). Note that
    there may be more than one staff per score part.

    Parameters
    ----------
    id : str
        The identifier of the part. In order to be compatible with
        MusicXML the identifier should not start with a number.
    part_name : str or None, optional
        Name for the part. Defaults to None
    part_abbreviation : str or None, optional
        Abbreviated name for part

    Attributes
    ----------
    id : str
        See parameters
    part_name : str
        See parameters
    part_abbreviation : str
        See parameters

    """

    def __init__(self, id, part_name=None, part_abbreviation=None):
        super().__init__()
        self.id = id
        self.parent = None
        self.part_name = part_name
        self.part_abbreviation = part_abbreviation

        # timeline init
        self._points = np.array([], dtype=TimePoint)
        self._quarter_times = [0]
        self._quarter_durations = [1]
        self._quarter_map = self.quarter_duration_map

    def __str__(self):
        return 'Part id="{}" name="{}"'.format(self.id, self.part_name)

    def _pp(self, tree):
        result = [self.__str__()]
        tree.push()
        N = len(self._points)
        for i, timepoint in enumerate(self._points):
            result.append('{}'.format(tree).rstrip())
            if i == N - 1:
                tree.last_item()
            else:
                tree.next_item()
            result.extend(timepoint._pp(tree))
        tree.pop()
        return result

    def pretty(self):
        """Return a pretty representation of this object.

        Returns
        -------
        str
            A pretty representation

        """
        return '\n'.join(self._pp(PrettyPrintTree()))

    @property
    def time_signature_map(self):
        """A function mapping timeline times to the beats and beat_type
        of the time signature at that time. The function can take
        scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        tss = np.array([(ts.start.t, ts.beats, ts.beat_type)
                        for ts in self.iter_all(TimeSignature)])
        if len(tss) == 0:
            # default time sig
            beats, beat_type = 4, 4
            LOGGER.warning('No time signatures found, assuming {}/{}'
                           .format(beats, beat_type))
            if self.first_point is None:
                t0, tN = 0, 0
            else:
                t0 = self.first_point.t
                tN = self.last_point.t
            tss = np.array([(t0, beats, beat_type),
                            (tN, beats, beat_type)])
        elif tss[0, 0] > self.first_point.t:
            tss = np.vstack(((self.first_point.t, tss[0, 1], tss[0, 2]),
                             tss))

        return interp1d(tss[:, 0], tss[:, 1:], axis=0, kind='previous',
                        bounds_error=False, fill_value='extrapolate')

    def _time_interpolator(self, quarter=False, inv=False):

        if len(self._points) < 2:
            return lambda x: np.zeros(len(x))

        keypoints = defaultdict(lambda: [None, None])
        _ = keypoints[self.first_point.t]
        _ = keypoints[self.last_point.t]
        for t, q in zip(self._quarter_times, self._quarter_durations):
            keypoints[t][0] = q
        if not quarter:
            for ts in self.iter_all(TimeSignature):
                # keypoints[ts.start.t][1] = int(np.log2(ts.beat_type))
                keypoints[ts.start.t][1] = ts.beat_type / 4
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
                keypoints_list.append([t] + kp)
        keypoints = np.array(keypoints_list, dtype=np.float)

        x = keypoints[:, 0]
        y = np.r_[0, np.cumsum((keypoints[:-1, 2]
                                * np.diff(keypoints[:, 0])) /
                               keypoints[:-1, 1])]

        m1 = next(self.first_point.iter_starting(Measure), None)

        if (m1 and m1.start is not None and m1.end is not None):

            f = interp1d(x, y)
            actual_dur = np.diff(f((m1.start.t, m1.end.t)))[0]
            ts = next(m1.start.iter_starting(TimeSignature), None)

            if ts:

                normal_dur = ts.beats
                if quarter:
                    normal_dur *= ts.beat_type / 4
                if actual_dur < normal_dur:
                    y -= actual_dur
            else:
                # warn
                pass

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
        return list(self.iter_all(Note, include_subclasses=True))

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
        return [note for note in self.iter_all(Note, include_subclasses=True)
                if note.tie_prev is None]

    def quarter_durations(self, start=None, end=None):
        """Return an Nx2 array with quarter duration (second column) and
        their respective times (first column).

        When a start and or end time is specified, the returned array
        will contain only the entries within those bounds.

        Parameters
        ----------
        start : number, optional
            Start of range
        end : number, optional
            End of range

        Returns
        -------
        ndarray
            An array with quarter durations and times

        """

        qd = np.column_stack((self._quarter_times, self._quarter_durations))
        if start is not None:
            qd = qd[qd[:, 0] >= start, :]
        if end is not None:
            qd = qd[qd[:, 0] < end, :]
        return qd

    @property
    def quarter_duration_map(self):
        """A function mapping timeline times to quarter durations in
        effect at those times. The function can take scalar values or
        lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        x = self._quarter_times
        y = self._quarter_durations
        if len(x) == 1:
            x = x + x
            y = y + y
        return interp1d(x, y, kind='previous',
                        bounds_error=False,
                        fill_value=(y[0], y[-1]))

    def set_quarter_duration(self, t, quarter):
        """Set the duration of a quarter note from timepoint `t`
        onwards.

        Setting the quarter note duration defines how intervals
        between timepoints are related to musical durations. For
        example when two timepoints `t1` and `t2` have associated
        times 10 and 20 respecively, then the interval between `t1`
        and `t2` corresponds to a half note when the quarter duration
        equals 5 during that interval.

        The quarter duration can vary throughout the part. When
        setting a quarter duration at time t, then that value takes
        effect until the time of the next quarter duration. If a
        different quarter duration was already set at time t, it wil
        be replaced.

        Note setting the quarter duration does not change the
        timepoints, only the relation to musical time. For
        illustration: in the example above, when changing the current
        quarter duration from 5 to 10, a note that starts at `t1` and
        ends at `t2` will change from being a half note to being a
        quarter note.

        Parameters
        ----------
        t : int
            Time at which to set the quarter duration
        quarter : int
            The quarter duration

        """

        # add quarter duration at time t, unless it is redundant. If another
        # quarter duration is at t, replace it.

        # shorthand
        times = self._quarter_times
        quarters = self._quarter_durations

        i = np.searchsorted(times, t)
        changed = False

        if (i == 0 or quarters[i - 1] != quarter):
            # add or replace
            if i == len(times) or times[i] != t:
                # add
                orig_quarter = 1 if i == 0 else quarters[i - 1]
                times.insert(i, t)
                quarters.insert(i, quarter)
                changed = True
            elif quarters[i] != quarter:
                # replace
                orig_quarter = quarters[i]
                quarters[i] = quarter
                changed = True
            else:
                # times[i] == t, quarters[i] == quarter
                pass

        if not changed:
            return

        if i + 1 == len(times):
            t_next = np.inf
        else:
            t_next = times[i + 1]

        # update quarter attribute of all timepoints in the range [t, t_next]
        start_idx = np.searchsorted(self._points, TimePoint(t))
        end_idx = np.searchsorted(self._points, TimePoint(t_next))
        for tp in self._points[start_idx:end_idx]:
            tp.quarter = quarter

        # update the interpolation function
        self._quarter_map = self.quarter_duration_map

    def _add_point(self, tp):
        # Add `TimePoint` object `tp` to the part, unless there is
        # already a timepoint at the same time.

        i = np.searchsorted(self._points, tp)
        if i == len(self._points) or self._points[i].t != tp.t:
            self._points = np.insert(self._points, i, tp)
            if i > 0:
                self._points[i - 1].next = self._points[i]
                self._points[i].prev = self._points[i - 1]
            if i < len(self._points) - 1:
                self._points[i].next = self._points[i + 1]
                self._points[i + 1].prev = self._points[i]

    def _remove_point(self, tp):
        i = np.searchsorted(self._points, tp)
        if self._points[i] == tp:
            self._points = np.delete(self._points, i)
            if i > 0:
                self._points[i - 1].next = self._points[i]
                self._points[i].prev = self._points[i - 1]
            if i < len(self._points) - 1:
                self._points[i].next = self._points[i + 1]
                self._points[i + 1].prev = self._points[i]

    def get_point(self, t):
        """Return the `TimePoint` object with time `t`, or None if there is no
        such object.

        """
        if t < 0:
            raise InvalidTimePointException('TimePoints should have non-negative integer values')

        i = np.searchsorted(self._points, TimePoint(t))
        if i < len(self._points) and self._points[i].t == t:
            return self._points[i]
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
        :class:`TimePoint`
            a TimePoint object with time `t`

        """
        if t < 0:
            raise InvalidTimePointException('TimePoints should have non-negative integer values')

        tp = self.get_point(t)
        if tp is None:
            tp = TimePoint(t, int(self._quarter_map(t)))
            self._add_point(tp)
        return tp

    def add(self, o, start=None, end=None):
        """Add an object to the timeline.

        An object can be added by start time, end time, or both,
        depending on which of the `start` and `end` keywords are
        provided. If neither is provided this method does nothing.

        `start` and `end` should be non-negative integers.

        Parameters
        ----------
        o : :class:`TimedObject`
            Object to be removed
        start : int, optional
            The start time of the object
        end : int, optional
            The end time of the object

        """
        if start is not None:
            if start < 0:
                raise InvalidTimePointException('TimePoints should have non-negative integer values')
            self.get_or_add_point(start).add_starting_object(o)
        if end is not None:
            if end < 0:
                raise InvalidTimePointException('TimePoints should have non-negative integer values')
            self.get_or_add_point(end).add_ending_object(o)

    def remove(self, o, which='both'):
        """Remove an object from the timeline.

        An object can be removed by start time, end time, or both.

        Parameters
        ----------
        o : :class:`TimedObject`
            Object to be removed
        which : {'start', 'end', 'both'}, optional
            Whether to remove o as a starting object, an ending
            object, or both. Defaults to 'both'.

        """

        if which in ('start', 'both') and o.start:
            try:
                o.start.starting_objects[o.__class__].remove(o)
            except:
                raise Exception('Not implemented: removing an object that is registered by its superclass')
            # cleanup timepoint if no starting/ending objects are left
            self._cleanup_point(o.start)
            o.start = None

        if which in ('end', 'both') and o.end:
            try:
                o.end.ending_objects[o.__class__].remove(o)
            except:
                raise Exception('Not implemented: removing an object that is registered by its superclass')
            # cleanup timepoint if no starting/ending objects are left
            self._cleanup_point(o.end)
            o.end = None

    def _cleanup_point(self, tp):
        # remove tp when it has no starting or ending objects
        if (sum(len(oo) for oo in tp.starting_objects.values()) +
            sum(len(oo) for oo in tp.ending_objects.values())) == 0:
            self._remove_point(o.end)

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
        start : :class:`TimePoint`, optional
            The start of the interval to search. If omitted or None,
            the search starts at the start of the timeline. Defaults
            to None.
        end : :class:`TimePoint`, optional
            The end of the interval to search. If omitted or None, the
            search ends at the end of the timeline. Defaults to None.
        include_subclasses : bool, optional
            If True also return instances that are subclasses of
            `cls`. Defaults to False.
        mode : {'starting', 'ending'}, optional
            Flag indicating whether to search for starting or ending
            objects. Defaults to 'starting'.

        Yields
        -------
        `cls`
            Instances of type `cls`

        """
        if not mode in ('starting', 'ending'):
            LOGGER.warning('unknown mode "{}", using "starting" instead'.format(mode))
            mode = 'starting'

        if start is None:
            start_idx = 0
        else:
            if not isinstance(start, TimePoint):
                start = TimePoint(start)
            start_idx = np.searchsorted(self._points, start)

        if end is None:
            end_idx = len(self._points)
        else:
            if not isinstance(end, TimePoint):
                end = TimePoint(end)
            end_idx = np.searchsorted(self._points, end)

        if mode == 'ending':
            for tp in self._points[start_idx: end_idx]:
                yield from tp.iter_ending(cls, include_subclasses)
        else:
            for tp in self._points[start_idx: end_idx]:
                yield from tp.iter_starting(cls, include_subclasses)


    @property
    def last_point(self):
        """The last TimePoint on the timeline, or None if the timeline is
        empty.

        Returns
        -------
        :class:`TimePoint`

        """
        return self._points[-1] if len(self._points) > 0 else None

    @property
    def first_point(self):
        """The first TimePoint on the timeline, or None if the timeline
        is empty.

        Returns
        -------
        :class:`TimePoint`
        """
        return self._points[0] if len(self._points) > 0 else None

    @property
    def note_array(self):
        """A structured array containing pitch, onset, duration, voice
        and id for each note

        """
        fields = [('onset', 'f4'),
                  ('duration', 'f4'),
                  ('pitch', 'i4'),
                  ('voice', 'i4'),
                  ('id', 'U256')]
        beat_map = self.beat_map
        note_array = []
        for note in self.notes_tied:
            note_on, note_off = beat_map([note.start.t, note.end.t])
            note_dur = note_off - note_on
            note_array.append((note_on, note_dur, note.midi_pitch,
                               note.voice, note.id))

        return np.array(note_array, dtype=fields)

    # @property
    # def part_names(self):
    #     # get instrument name parts recursively
    #     chunks = []

    #     if self.part_name is not None:
    #         chunks.append(self.part_name)
    #         yield self.part_name

    #     pg = self.parent
    #     while pg is not None:
    #         if pg.group_name is not None:
    #             chunks.insert(0, pg.group_name)
    #             yield '  '.join(chunks)
    #         pg = pg.parent


class TimePoint(ComparableMixin):

    """A TimePoint represents a temporal position within a
    :class:`Part`.

    TimePoints are used to keep track of the starting and ending of
    musical elements in the part. They are created automatically when
    adding musical elements to a part using its :meth:`~Part.add`
    method, so there should be normally no reason to instantiate
    TimePoints manually.

    Parameters
    ----------
    t : int
        The time associated to this TimePoint. Should be a non-
        negative integer.
    quarter : int
        The duration of a quarter note at this TimePoint

    Attributes
    ----------
    t : int
        See parameters
    quarter : int
        See parameters
    starting_objects : dictionary
        A dictionary where the musical objects starting at this time
        are grouped by class.
    ending_objects : dictionary
        A dictionary where the musical objects ending at this time are
        grouped by class.
    prev : TimePoint
        The preceding TimePoint (or None if there is none)
    next : TimePoint
        The succeding TimePoint (or None if there is none)

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
        return ('TimePoint t={} quarter={}'
                .format(self.t, self.quarter))

    def add_starting_object(self, obj):
        """Add object `obj` to the list of starting objects.

        """
        obj.start = self
        self.starting_objects[type(obj)].append(obj)

    def remove_starting_object(self, obj):
        """Remove object `obj` from the list of starting objects.

        """
        # TODO: check if object is stored under a superclass
        obj.start = None
        if type(obj) in self.starting_objects:
            try:
                self.starting_objects[type(obj)].remove(obj)
            except ValueError:
                # don't complain if the object isn't in starting_objects
                pass

    def remove_ending_object(self, obj):
        """Remove object `obj` from the list of ending objects.

        """
        # TODO: check if object is stored under a superclass
        obj.end = None
        if type(obj) in self.ending_objects:
            try:
                self.ending_objects[type(obj)].remove(obj)
            except ValueError:
                # don't complain if the object isn't in ending_objects
                pass

    def add_ending_object(self, obj):
        """Add object `obj` to the list of ending objects.

        """
        obj.end = self
        self.ending_objects[type(obj)].append(obj)

    def iter_starting(self, cls, include_subclasses=False):
        """Iterate over all objects of type `cls` that start at this
        time point.

        Parameters
        ----------
        cls : class
            The type of objects to iterate over
        include_subclasses : bool, optional
            When True, include all objects of all subclasses of `cls`
            in the iteration. Defaults to False.

        Yields
        -------
        cls
            Instance of type `cls`

        """
        yield from self.starting_objects[cls]
        if include_subclasses:
            for subcls in iter_subclasses(cls):
                yield from self.starting_objects[subcls]

    def iter_ending(self, cls, include_subclasses=False):
        """Iterate over all objects of type `cls` that end at this
        time point.

        Parameters
        ----------
        cls : class
            The type of objects to iterate over
        include_subclasses : bool, optional
            When True, include all objects of all subclasses of `cls`
            in the iteration. Defaults to False.

        Yields
        -------
        cls
            Instance of type `cls`

        """
        yield from self.ending_objects[cls]
        if include_subclasses:
            for subcls in iter_subclasses(cls):
                yield from self.ending_objects[subcls]

    def iter_prev(self, cls, eq=False, include_subclasses=False):
        """Iterate backwards in time from the current timepoint over
        starting object(s) of type `cls`.

        Parameters
        ----------
        cls : class
            Class of objects to iterate over
        eq : bool, optional
            If True start iterating at the current timepoint, rather
            than its predecessor. Defaults to False.
        include_subclasses : bool, optional
            If True include subclasses of `cls` in the iteration.
            Defaults to False.

        Yields
        ------
        cls
            Instances of `cls`

        """
        if eq:
            tp = self
        else:
            tp = self.prev

        while tp:
            yield from tp.iter_starting(cls, include_subclasses)
            tp = tp.prev

    def iter_next(self, cls, eq=False, include_subclasses=False):
        """Iterate forwards in time from the current timepoint over
        starting object(s) of type `cls`.

        Parameters
        ----------
        cls : class
            Class of objects to iterate over
        eq : bool, optional
            If True start iterating at the current timepoint, rather
            than its successor. Defaults to False.
        include_subclasses : bool, optional
            If True include subclasses of `cls` in the iteration.
            Defaults to False.

        Yields
        ------
        cls
            Instances of `cls`

        """
        if eq:
            tp = self
        else:
            tp = self.next

        while tp:
            yield from tp.iter_starting(cls, include_subclasses)
            tp = tp.next


    def _cmpkey(self):
        # This method returns the value to be compared (code for that is in
        # the ComparableMixin class)
        return self.t

    def _pp(self, tree):
        # pretty print the timepoint, including its starting and ending
        # objects
        result = ['{}{}'.format(tree, self.__str__())]
        tree.push()

        ending_items_lists = sorted_dict_items(self.ending_objects.items(),
                                               key=lambda x: x[0].__name__)
        starting_items_lists = sorted_dict_items(self.starting_objects.items(),
                                                 key=lambda x: x[0].__name__)

        ending_items = [o for _, oo in ending_items_lists for o in oo]
        starting_items = [o for _, oo in starting_items_lists for o in oo]

        if ending_items:

            result.append('{}'.format(tree).rstrip())

            if starting_items:
                tree.next_item()
            else:
                tree.last_item()

            result.append('{}ending objects'.format(tree))
            tree.push()
            result.append('{}'.format(tree).rstrip())

            for i, item in enumerate(ending_items):

                if i == (len(ending_items) - 1):
                    tree.last_item()
                else:
                    tree.next_item()

                result.append('{}{}'.format(tree, item))

            tree.pop()

        if starting_items:

            result.append('{}'.format(tree).rstrip())
            tree.last_item()
            result.append('{}starting objects'.format(tree))
            tree.push()
            result.append('{}'.format(tree).rstrip())

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
    """This is the base class of all classes that have a start and end
    point. The start and end attributes initialized to None, and are
    set/unset when the object is added to/removed from a Part, using
    its :meth:`~Part.add` and :meth:`~Part.remove` methods, respectively.

    Attributes
    ----------
    start : :class:`TimePoint`
        Start time of the object
    end : :class:`TimePoint`
        End time of the object

    """
    def __init__(self):
        super().__init__()
        self.start = None
        self.end = None


class Page(TimedObject):
    """A page in a musical score. Its start and end times describe the
    range of musical time that is spanned by the page.

    Parameters
    ----------
    number : int, optional
        The number of the system. Defaults to 0.

    Attributes
    ----------
    number : int
        See parameters

    """
    def __init__(self, number=0):
        super().__init__()
        self.number = number

    def __str__(self):
        return 'Page number={}'.format(self.number)


class System(TimedObject):
    """A system in a musical score. Its start and end times describe the
    range of musical time that is spanned by the system.

    Parameters
    ----------
    number : int, optional
        The number of the system. Defaults to 0.

    Attributes
    ----------
    number : int
        See parameters

    """

    def __init__(self, number=0):
        super().__init__()
        self.number = number

    def __str__(self):
        return 'System number={}'.format(self.number)


class Clef(TimedObject):
    """A clef.

    Clefs associate the lines of a staff to musical pitches.

    Parameters
    ----------
    number : int, optional
        The number of the staff to which this clef belongs.
    sign : {'G', 'F', 'C', 'percussion', 'TAB', 'jianpu',  'none'}
        The sign of the clef
    line : int
        The staff line at which the sign is positioned
    octave_change : int
        The number of octaves to shift the pitches up (postive) or
        down (negative)

    Attributes
    ----------
    nr : int
        See parameters
    sign : {'G', 'F', 'C', 'percussion', 'TAB', 'jianpu',  'none'}
        See parameters
    line : int
        See parameters
    octave_change : int
        See parameters

    """

    def __init__(self, number, sign, line, octave_change):

        super().__init__()
        self.number = number
        self.sign = sign
        self.line = line
        self.octave_change = octave_change

    def __str__(self):
        return 'Clef sign={} line={} number={}'.format(self.sign, self.line, self.number)


class Slur(TimedObject):
    """A slur.

    Slurs indicate musical grouping across notes.

    Parameters
    ----------
    start_note : :class:`Note`, optional
        The note at which this slur starts. Defaults to None.
    end_note : :class:`Note`, optional
        The note at which this slur ends. Defaults to None.

    Attributes
    ----------
    start_note : :class:`Note` or None
        See parameters
    end_note : :class:`Note` or None
        See parameters


    """

    def __init__(self, start_note=None, end_note=None):
        super().__init__()
        self._start_note = None
        self._end_note = None
        self.start_note = start_note
        self.end_note= end_note
        # maintain a list of attributes to update when cloning this instance
        self._ref_attrs.extend(['start_note', 'end_note'])

    @property
    def start_note(self):
        return self._start_note

    @start_note.setter
    def start_note(self, note):
        # make sure we received a note
        if note:
            if note.start:
                #  remove the slur from the current start time
                if self.start_note and self.start_note.start:
                    self.start_note.start.remove_starting_object(self)
            # else:
            #     LOGGER.warning('Note has no start time')
            note.slur_starts.append(self)
        self._start_note = note

    @property
    def end_note(self):
        return self._end_note

    @end_note.setter
    def end_note(self, note):
        # make sure we received a note
        if note:
            if note.end:
                if self.end_note and self.end_note.end:
                    #  remove the slur from the currentend time
                    self.end_note.end.remove_ending_object(self)
            # else:
            #     LOGGER.warning('Note has no end time')
            note.slur_stops.append(self)
        self._end_note = note

    def __str__(self):
        # return 'slur at voice {0} (ends at {1})'.format(self.voice, self.end and self.end.t)
        start = '' if self.start_note is None else 'start={}'.format(self.start_note.id)
        end = '' if self.end_note is None else 'end={}'.format(self.end_note.id)
        return ' '.join(('Slur', start, end)).strip()


class Tuplet(TimedObject):
    """A tuplet.

    Tuplets indicate musical grouping across notes.

    Parameters
    ----------
    start_note : :class:`Note`, optional
        The note at which this tuplet starts. Defaults to None.
    end_note : :class:`Note`, optional
        The note at which this tuplet ends. Defaults to None.

    Attributes
    ----------
    start_note : :class:`Note` or None
        See parameters
    end_note : :class:`Note` or None
        See parameters


    """

    def __init__(self, start_note=None, end_note=None):
        super().__init__()
        self._start_note = None
        self._end_note = None
        self.start_note = start_note
        self.end_note= end_note
        # maintain a list of attributes to update when cloning this instance
        self._ref_attrs.extend(['start_note', 'end_note'])

    @property
    def start_note(self):
        return self._start_note

    @start_note.setter
    def start_note(self, note):
        # make sure we received a note
        if note:
            if note.start:
                #  remove the tuplet from the current start time
                if self.start_note and self.start_note.start:
                    self.start_note.start.remove_starting_object(self)
            # else:
            #     LOGGER.warning('Note has no start time')
            note.tuplet_starts.append(self)
        self._start_note = note

    @property
    def end_note(self):
        return self._end_note

    @end_note.setter
    def end_note(self, note):
        # make sure we received a note
        if note:
            if note.end:
                if self.end_note and self.end_note.end:
                    #  remove the tuplet from the currentend time
                    self.end_note.end.remove_ending_object(self)
            # else:
            #     LOGGER.warning('Note has no end time')
            note.tuplet_stops.append(self)
        self._end_note = note


    def __str__(self):
        start = '' if self.start_note is None else 'start={}'.format(self.start_note.id)
        end = '' if self.end_note is None else 'end={}'.format(self.end_note.id)
        return ' '.join(('Tuplet', start, end)).strip()


class Repeat(TimedObject):
    """A repeat.

    This class represents a repeated section in the score, designated
    by its start and end times.

    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Repeat (from {0} to {1})'.format(self.start and self.start.t, self.end and self.end.t)


class DaCapo(TimedObject):
    """A Da Capo sign.

    """

    def __str__(self):
        return u'Dacapo'


class Fine(TimedObject):
    """A Fine sign.

    """

    def __str__(self):
        return 'Fine'


class Fermata(TimedObject):
    """A Fermata sign.

    Parameters
    ----------
    ref : :class:`TimedObject` or None, optional
        An object to which this fermata applies. In practice this is a
        Note or a Barline. Defaults to None.

    Attributes
    ----------
    ref : :class:`TimedObject` or None
        See parameters

    """

    def __init__(self, ref=None):
        super().__init__()
        # ref(erent) can be a note or a barline
        self.ref = ref

    def __str__(self):
        return 'Fermata ref={}'.format(self.ref)


class Ending(TimedObject):

    """Class that represents one part of a 1---2--- type ending of a musical
    passage (a.k.a Volta brackets).

    Parameters
    ----------
    number : int
        The number associated to this ending

    Attributes
    ----------
    number : int
        See parameters

    """

    def __init__(self, number):
        super().__init__()
        self.number = number

    def __str__(self):
        return 'Ending (from {0} to {1})'.format(self.start.t, self.end.t)


class Measure(TimedObject):

    """A measure.

    Parameters
    ----------
    number : int or None, optional
        The number of the measure. Defaults to None

    Attributes
    ----------
    number : int
        See parameters

    """

    def __init__(self, number=None):
        super().__init__()
        self.number = number

    def __str__(self):
        # return 'Measure {0} at page {1}, system {2}'.format(self.number, self.page, self.system)
        return 'Measure number={0}'.format(self.number)

    @property
    def page(self):
        """The page number on which this measure appears, or None if
        there is no associated page.

        Returns
        -------
        int or None

        """
        page = next(self.start.iter_prev(Page, eq=True), None)
        if page:
            return page.number
        else:
            return None

    @property
    def system(self):
        """The system number in which this measure appears, or None if
        there is no associated system.

        Returns
        -------
        int or None

        """
        system = next(self.start.iter_prev(System, eq=True), None)
        if system:
            return system.number
        else:
            return None

    # TODO: add `incomplete` or `anacrusis` property


class TimeSignature(TimedObject):
    """A time signature.

    Parameters
    ----------
    beats : int
        The number of beats in a measure
    beat_type : int
        The note type that defines the beat unit. (4 for quarter
        notes, 2 for half notes, etc.)

    Attributes
    ----------
    beats : int
        See parameters
    beat_type : int
        See parameters

    """

    def __init__(self, beats, beat_type):
        super().__init__()
        self.beats = beats
        self.beat_type = beat_type

    def __str__(self):
        return 'TimeSignature {0}/{1}'.format(self.beats, self.beat_type)


class Tempo(TimedObject):
    """A tempo indication.

    Parameters
    ----------
    bpm : number
        The tempo indicated in rate per minute
    unit : str or None, optional
        The unit to which the specified rate correspnds. This is a
        string that expreses a duration category, such as "q" for
        quarter "h." for dotted half, and so on. When None, the unit
        is assumed to be quarters. Defaults to None.

    Attributes
    ----------
    bpm : number
        See parameters
    unit : str or None
        See parameters

    """
    def __init__(self, bpm, unit=None):
        super().__init__()
        self.bpm = bpm
        self.unit = unit

    @property
    def microseconds_per_quarter(self):
        """The number of microseconds per quarter under this tempo.

        This is useful for MIDI representations.

        Returns
        -------
        int

        """
        return int(np.round(60 * (10**6 / to_quarter_tempo(self.unit or 'q', self.bpm))))

    def __str__(self):
        if self.unit:
            return 'Tempo {}={}'.format(self.unit, self.bpm)
        else:
            return 'Tempo bpm={0}'.format(self.bpm)


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
    fifths : number
        See parameters
    mode : str
        See parameters

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
        return ('Key signature fifths={}, mode={} ({})'
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

    Attributes
    ----------
    diatonic : int
        See parameters
    chromatic : int
        See parameters

    """

    def __init__(self, diatonic, chromatic):
        super().__init__()
        self.diatonic = diatonic
        self.chromatic = chromatic

    def __str__(self):
        return 'Transposition diatonic={0}, chromatic={1}'.format(self.diatonic, self.chromatic)


class Words(TimedObject):
    """A textual element in the score.

    Parameters
    ----------
    text : str
        The text
    staff : int or None, optional
        The staff to which the text is associated. Defaults to None

    Attributes
    ----------
    text : str
        See parameters
    staff : int or None, optional
        See parameters

    """

    def __init__(self, text, staff=None):
        super().__init__()
        self.text = text
        self.staff = staff

    def __str__(self):
        return '{} "{}"'.format(type(self).__name__, self.text)


class Direction(TimedObject):
    """Base class for performance directions in the score.
    
    """

    def __init__(self, text, raw_text=None, staff=None):
        super().__init__()
        self.text = text
        self.raw_text = raw_text
        self.staff = staff

    def __str__(self):
        if self.raw_text is not None:
            return '{} "{}" raw_text="{}"'.format(type(self).__name__, self.text, self.raw_text)
        else:
            return '{} "{}"'.format(type(self).__name__, self.text)



class LoudnessDirection(Direction): pass
class TempoDirection(Direction): pass
class ArticulationDirection(Direction): pass

class ConstantDirection(Direction): pass
class DynamicDirection(Direction): pass
class ImpulsiveDirection(Direction): pass

class ConstantLoudnessDirection(ConstantDirection, LoudnessDirection): pass
class ConstantTempoDirection(ConstantDirection, TempoDirection): pass
class ConstantArticulationDirection(ConstantDirection, ArticulationDirection): pass

class DynamicLoudnessDirection(DynamicDirection, LoudnessDirection):
    def __init__(self, *args, wedge=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.wedge = wedge
    def __str__(self):
        if self.wedge:
            return '{} wedge'.format(super().__str__())
        else:
            return super().__init__()

class DynamicTempoDirection(DynamicDirection, TempoDirection): pass

class IncreasingLoudnessDirection(DynamicLoudnessDirection): pass
class DecreasingLoudnessDirection(DynamicLoudnessDirection): pass
class IncreasingTempoDirection(DynamicTempoDirection): pass
class DecreasingTempoDirection(DynamicTempoDirection): pass

class ImpulsiveLoudnessDirection(ImpulsiveDirection, LoudnessDirection): pass

class ResetTempoDirection(ConstantTempoDirection):
    @property
    def reference_tempo(self):
        direction = None
        for d in self.start.iter_prev(ConstantTempoDirection):
            direction = d
        return direction


class GenericNote(TimedObject):
    """Represents the common aspects of notes and rests (and in the future
    unpitched notes)

    Parameters
    ----------
    voice : integer, optional (default: None)
    id : integer, optional (default: None)

    """

    def __init__(self, id=None, voice=None, staff=None, symbolic_duration=None, articulations=None):
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
        self.tuplet_stops = []
        self.tuplet_starts = []

        # maintain a list of attributes to update when cloning this instance
        self._ref_attrs.extend(['tie_prev', 'tie_next',
                                'slur_stops', 'slur_starts',
                                'tuplet_stops', 'tuplet_starts'])

    @property
    def symbolic_duration(self):
        """TODO

        Parameters
        ----------

        Returns
        -------
        type
            Description of return value
        """

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
        """TODO

        Parameters
        ----------

        Returns
        -------
        type
            Description of return value
        """

        if self.symbolic_duration:
            return symbolic_to_numeric_duration(self.symbolic_duration, self.start.quarter)
        else:
            return None

    @property
    def tie_prev_notes(self):
        """TODO

        Parameters
        ----------

        Returns
        -------
        type
            Description of return value
        """

        if self.tie_prev:
            return self.tie_prev.tie_prev_notes + [self.tie_prev]
        else:
            return []

    @property
    def tie_next_notes(self):
        """TODO

        Parameters
        ----------

        Returns
        -------
        type
            Description of return value
        """

        if self.tie_next:
            return [self.tie_next] + self.tie_next.tie_next_notes
        else:
            return []

    def iter_voice_prev(self):
        """TODO

        Parameters
        ----------

        Returns
        -------
        type
            Description of return value
        """

        for n in self.start.iter_prev(GenericNote, include_subclasses=True):
            if n.voice == n.voice:
                yield n

    def iter_voice_next(self):
        """TODO

        Parameters
        ----------

        Returns
        -------
        type
            Description of return value
        """

        for n in self.start.iter_next(GenericNote, include_subclasses=True):
            if n.voice == n.voice:
                yield n

    def iter_chord(self, same_duration=True, same_voice=True):
        """TODO

        Parameters
        ----------
        same_duration: type, optional
            Description of `same_duration`
        same_voice: type, optional
            Description of `same_voice`

        Returns
        -------
        type
            Description of return value
        """

        for n in self.start.iter_starting(GenericNote, include_subclasses=True):
            if (((not same_voice) or n.voice == self.voice)
                    and ((not same_duration) or (n.duration == self.duration))):
                yield n

    def __str__(self):
        s = ('{} id={} voice={} staff={} type={}'
             .format(type(self).__name__, self.id, self.voice, self.staff,
                     format_symbolic_duration(self.symbolic_duration)))
        if self.articulations:
            s += ' articulations=({})'.format(", ".join(self.articulations))
        if self.tie_prev or self.tie_next:
            all_tied = self.tie_prev_notes + [self] + self.tie_next_notes
            tied_dur = '+'.join(format_symbolic_duration(n.symbolic_duration) for n in all_tied)
            tied_id = '+'.join(n.id or 'None' for n in all_tied)
            return s + ' tie_group={}'.format(tied_id)
        else:
            return s


class Note(GenericNote):
    def __init__(self, step, octave, alter=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = step
        self.octave = octave
        self.alter = alter

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
        return pitch_spelling_to_midi_pitch(step=self.step,
                                            octave=self.octave,
                                            alter=self.alter)

    # TODO: include morphetic pitch method based in code in musicanalysis package

    @property
    def alter_sign(self):
        """The alteration of the note

        Returns
        -------
        str

        """
        return ALTER_SIGNS[self.alter]


class Rest(GenericNote):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GraceNote(Note):
    def __init__(self, grace_type, *args, steal_proportion=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grace_type = grace_type
        self.steal_proportion = steal_proportion
        self.grace_next = None
        self.grace_prev = None
        self._ref_attrs.extend(['grace_next', 'grace_prev'])

    @property
    def main_note(self):
        n = self.grace_next
        while isinstance(n, GraceNote):
            n = n.grace_next
        return n

    @property
    def grace_seq_len(self):
        return (sum(1 for _ in self.iter_grace_seq(backwards=True))
                + sum(1 for _ in self.iter_grace_seq())
                - 1) # subtract one because self is counted twice

    def iter_grace_seq(self, backwards=False):
        """Iterate over this and all subsequent/preceding grace notes,
        excluding the main note.

        Parameters
        ----------
        backwards : bool, optional
            When True, iterate over preceding grace notes. Otherwise
            iterate over subsequent grace notes. Defaults to False.

        Yields
        ------
        GraceNote
        
        """

        yield self
        if backwards:
            n = self.grace_prev
        else:
            n = self.grace_next
        while isinstance(n, GraceNote):
            yield n
            if backwards:
                n = n.grace_prev
            else:
                n = n.grace_next

    def __str__(self):
        s = ' '.join(
            (super().__str__(),
             'main_note={}'.format(self.main_note and self.main_note.id)))
             # 'grace_prev={}'.format(self.grace_prev.id if self.grace_prev else None),
             # 'grace_next={}'.format(self.grace_next.id if self.grace_next else None)),
        return s

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

    name : str or None

    number : int

    parent : PartGroup or None

    children : list of Part or PartGroup objects


    """

    def __init__(self, group_symbol=None, group_name=None, number=None):
        self.group_symbol = group_symbol
        self.group_name = group_name
        self.number = number
        self.parent = None
        self.children = []

    def _pp(self, tree):
        result = ['{}PartGroup: group_name="{}" group_symbol="{}"'
                  .format(tree, self.group_name, self.group_symbol)]
        tree.push()
        N = len(self.children)
        for i, child in enumerate(self.children):
            result.append('{}'.format(tree).rstrip())
            if i == N - 1:
                tree.last_item()
            else:
                tree.next_item()
            result.extend(child._pp(tree))
        tree.pop()
        return result

    def pretty(self):
        """Return a pretty representation of this object.

        Returns
        -------
        str
            A pretty representation

        """
        return '\n'.join(self._pp(PrettyPrintTree()))


class ScoreVariant(object):
    """
    """

    def __init__(self, part, start_time=0):
        self.t_unfold = start_time
        self.segments = []
        self.part = part

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
        return 'Segment {}'.format(self.segment_times)

    def clone(self):
        """
        Return a clone of the ScoreVariant
        """
        clone = ScoreVariant(self.part, self.t_unfold)
        clone.segments = self.segments[:]
        return clone

    def create_variant_part(self):
        part = Part(self.part.id, part_name=self.part.part_name)

        for start, end, offset in self.segments:
            delta = offset - start.t
            qd = self.part.quarter_durations(start.t, end.t)
            for t, quarter in qd:
                part.set_quarter_duration(t + delta, quarter)
            # After creating the new part we need to replace references to
            # objects in the old part to references in the new part
            # (e.g. t.next, t.prev, note.tie_next). For this we keep track of
            # correspondences between objects (timepoints, notes, measures,
            # etc), in o_map
            o_map = {}
            o_new = set()
            tp = start
            while tp != end:
                # make a new timepoint, corresponding to tp
                tp_new = part.get_or_add_point(tp.t + delta)
                o_gen = (o for oo in tp.starting_objects.values() for o in oo)
                for o in o_gen:

                    # special cases:

                    # don't include repeats/endings in the unfolded part
                    if isinstance(o, (Repeat, Ending)):
                        continue
                    # don't repeat time sig if it hasn't changed
                    elif isinstance(o, TimeSignature):
                        prev = next(tp_new.iter_prev(TimeSignature), None)
                        if ((prev is not None)
                            and ((o.beats, o.beat_type) == (prev.beats, prev.beat_type))):
                            continue
                    # don't repeat key sig if it hasn't changed
                    elif isinstance(o, KeySignature):
                        prev = next(tp_new.iter_prev(KeySignature), None)
                        if ((prev is not None)
                            and ((o.fifths, o.mode) == (prev.fifths, prev.mode))):
                            continue

                    # make a copy of the object
                    o_copy = copy(o)
                    # add it to the set of new objects (for which the refs will be replaced)
                    o_new.add(o_copy)
                    # keep track of the correspondence between o and o_copy
                    o_map[o] = o_copy
                    # add the start of the new object to the part
                    tp_new.add_starting_object(o_copy)
                    if o.end is not None:
                        # add the end of the object to the part
                        tp_end = part.get_or_add_point(o.end.t + delta)
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
                    tp_new = part.get_or_add_point(end.t + delta)
                    tp_new.add_starting_object(o_copy)

            # for each of the new objects, replace the references to the old
            # objects to their corresponding new objects
            for o in o_new:
                o.replace_refs(o_map)

        # replace prev/next references in timepoints
        for tp, tp_next in iter_current_next(part._points):
            tp.next = tp_next
            tp_next.prev = tp

        return part


def iter_unfolded_parts(part):
    """TODO: Description

    Parameters
    ----------
    part: type
        Description of `part`

    Yields
    ------
    ScoreVariant
        Description of return value
    """

    for sv in make_score_variants(part):
        yield sv.create_variant_part()


def unfold_part_maximal(part):
    """Return the "maximally" unfolded part, that is, a copy of the
    part where all segments marked with repeat signs are included
    twice.

    Returns
    -------
    Part
        The unfolded Part

    """

    sv = make_score_variants(part)[-1]
    return sv.create_variant_part()


def make_score_variants(part):
    # non-public (use unfold_part_maximal, or iter_unfolded_parts)
    """Create a list of ScoreVariant objects, each representing a
    distinct way to unfold the score, based on the repeat structure.

    Parameters
    ----------
    part: Part
        A part for which to make the score variants

    Returns
    -------
    list
        List of ScoreVariant objects

    Notes
    -----
    This function does not currently support nested repeats, such as in
    case 45d of the MusicXML Test Suite.

    """

    if len(list(part.iter_all(DaCapo)) +
           list(part.iter_all(Fine))) > 0:
        LOGGER.warning(('Generation of repeat structures involving da '
                        'capo/fine/coda/segno directions is not '
                        'supported yet'))

    # TODO: check if we need to wrap in list
    repeats = list(part.iter_all(Repeat))
    # repeats may not have start or end times. `repeats_to_start_end`
    # returns the start/end paisr for each repeat, making educated guesses
    # when these are missing.
    repeat_start_ends = repeats_to_start_end(repeats,
                                             part.first_point,
                                             part.last_point)

    # check for nestings and raise if necessary
    if any(n < c for c, n in iter_current_next(repeat_start_ends)):
        raise NotImplementedError('Nested endings are currently not supported')

    # t_score is used to keep the time in the score
    t_score = part.first_point
    svs = [ScoreVariant(part)]
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

                    # new_sv includes the 1/2 ending repeat, which means:
                    # 1. from repeat start to repeat end (which includes ending 1)
                    new_sv.add_segment(rep_start, rep_end)
                    # 2. from repeat start to ending 1 start
                    new_sv.add_segment(rep_start, ending1.start)
                    # 3. ending 2 start to ending 2 end
                    new_sv.add_segment(ending2.start, ending2.end)
    
                    # new score time will be the score time
                    t_end = ending2.end
    
                else:
                    # ending 1 without ending 2, should not happen normally
                    LOGGER.warning('ending 1 without ending 2')
                    # new score time will be the score time
                    t_end = ending1.end
            else:
                # add the first occurrence of the repeat
                sv.add_segment(rep_start, rep_end)

                # no: add the full interval of the repeat (the second time)
                new_sv.add_segment(rep_start, rep_end)
                new_sv.add_segment(rep_start, rep_end)

                # update the score time
                t_end = rep_end

            # add both score variants
            new_svs.append(sv)
            new_svs.append(new_sv)
        t_score = t_end
        
        svs = new_svs

    # are we at the end of the piece already?
    if t_score < part.last_point:
        # no, append the interval from the current score
        # position to the end of the piece
        for sv in svs:
            sv.add_segment(t_score, part.last_point)

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

    timesigs = np.array([(ts.start.t, ts.beats)
                         for ts in part.iter_all(TimeSignature)],
                        dtype=np.int)

    if len(timesigs) == 0:
        LOGGER.warning('No time signatures found, not adding measures')
        return
    
    start = part.first_point.t
    end = part.last_point.t

    if start == end:
        return
    
    # make sure we cover time from the start of the timeline
    if len(timesigs) == 0 or timesigs[0, 0] > start:
        timesigs = np.vstack(([[start, 4]], timesigs))

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
            measure_end_beats = min(beat_map(pos) + measure_dur, beat_map(end))
            measure_end = min(ts_end, inv_beat_map(measure_end_beats))

            # any existing measures between measure_start and measure_end
            existing_measure = next(part.iter_all(Measure, measure_start, measure_end), None)

            if existing_measure:
                if existing_measure.start.t == measure_start:
                    assert existing_measure.end.t > pos
                    pos += existing_measure.end.t
                    existing_measure.number = mcounter
                    mcounter += 1
                    continue
                else:
                    measure_end = existing_measure.start.t

            part.add(Measure(number=mcounter), int(measure_start), int(measure_end))

            if existing_measure:
                pos = existing_measure.end.t
                existing_measure.number = mcounter + 1
                mcounter = mcounter + 2
            else:
                pos = measure_end
                mcounter += 1


def remove_grace_notes(part):
    """Remove all grace notes from a timeline.

    The specified timeline object will be modified in place.

    Parameters
    ----------
    timeline : Timeline
        The timeline from which to remove the grace notes

    """
    # for point in part.points:
    #     point.starting_objects[Note] = [n for n in point.starting_objects[Note]
    #                                     if n.grace_type is None]
    #     point.ending_objects[Note] = [n for n in point.ending_objects[Note]
    #                                   if n.grace_type is None]
    for gn in list(part.iter_all(GraceNote)):
        part.remove(gn)

def expand_grace_notes(part):
    """Expand grace note durations in a part.

    The specified part object will be modified in place.

    Parameters
    ----------
    part : Part
        The part on which to expand the grace notes

    """
    for gn in part.iter_all(GraceNote):
        dur = symbolic_to_numeric_duration(gn.symbolic_duration, gn.start.quarter)
        part.remove(gn, 'end')
        part.add(gn, end=gn.start.t+int(np.round(dur)))


def iter_parts(partlist):
    """Iterate over all Part instances in partlist, which is a list of
    either Part or PartGroup instances. PartGroup instances contain
    one or more parts or further partgroups, and are traversed in a
    depth-first fashion.

    This function is designed to take the result of
    :func:`partitura.load_score_midi` and :func:`partitura.load_musicxml` as
    input.

    Parameters
    ----------
    partlist : list, Part, or PartGroup
        A :class:`partitura.score.Part` object,
        :class:`partitura.score.PartGroup` or a list of these

    Yields
    -------
        `Part` instances in `partlist`

    """

    if not isinstance(partlist, (list, tuple, set)):
        partlist = [partlist]

    for el in partlist:
        if isinstance(el, Part):
            yield el
        else:
            for eel in iter_parts(el.children):
                yield eel


def repeats_to_start_end(repeats, first, last):
    # non-public
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


def _make_tied_note_id(prev_id):
    # non-public
    """Create a derived note ID for newly created notes, by appending
    letters to the ID. If the original ID has the form X-Y (e.g.
    n1-1), then the letter will be appended to the X part.

    Parameters
    ----------
    prev_id : str
        Original note ID

    Returns
    -------
    str
        Derived note ID

    Examples
    --------
    >>> _make_tied_note_id('n0')
    'n0a'
    >>> _make_tied_note_id('n0a')
    'n0b'
    >>> _make_tied_note_id('n0-1')
    'n0a-1'

    """
    prev_id_parts = prev_id.split('-', 1)
    prev_id_p1 = prev_id_parts[0]
    if prev_id_p1:
        if ord(prev_id_p1[-1]) < ord('a') - 1:
            return '-'.join(['{}a'.format(prev_id_p1)] +
                            prev_id_parts[1:])
        else:
            return '-'.join(['{}{}'.format(prev_id_p1[:-1], chr(ord(prev_id[-1]) + 1))]
                            + prev_id_parts[1:])
    else:
        return None


def tie_notes(part):
    """TODO: Description

    Parameters
    ----------
    part: type
        Description of `part`

    """

    # split and tie notes at measure boundaries
    for note in part.iter_all(Note):
        next_measure = next(note.start.iter_next(Measure), None)
        cur_note = note
        note_end = cur_note.end

        # keep the list of stopping slurs, we need to transfer them to the last
        # tied note
        slur_stops = cur_note.slur_stops

        while next_measure and cur_note.end > next_measure.start:
            part.remove(cur_note, 'end')
            cur_note.slur_stops = []
            part.add(cur_note, None, next_measure.start.t)
            cur_note.symbolic_duration = estimate_symbolic_duration(next_measure.start.t - cur_note.start.t, cur_note.start.quarter)
            sym_dur = estimate_symbolic_duration(note_end.t - next_measure.start.t, next_measure.start.quarter)
            if cur_note.id is not None:
                note_id = _make_tied_note_id(cur_note.id)
            else:
                note_id = None
            next_note = Note(note.step, note.octave, note.alter, id=note_id,
                             voice=note.voice, staff=note.staff,
                             symbolic_duration=sym_dur)
            part.add(next_note, next_measure.start.t, note_end.t)

            cur_note.tie_next = next_note
            next_note.tie_prev = cur_note

            cur_note = next_note

            next_measure = next(cur_note.start.iter_next(Measure), None)

        if cur_note != note:
            for slur in slur_stops:
                slur.end_note = cur_note

    # then split/tie any notes that do not have a fractional/dot duration
    divs_map = part.quarter_duration_map
    notes = part.iter_all(Note)

    max_splits = 3
    failed = 0
    succeeded = 0
    for i, note in enumerate(notes):
        if note.symbolic_duration is None:

            splits = find_tie_split(note.start.t, note.end.t, int(divs_map(note.start.t)), max_splits)

            if splits:
                succeeded += 1
                split_note(part, note, splits)
            else:
                failed += 1
    # print(failed, succeeded, failed/succeeded)


def set_end_times(parts):
    """
    Set missing end times of musical elements in a part to equal the start times
    of the subsequent element of the same class. This is useful for some classes

    Parameters
    ----------
    part: Part or PartGroup, or list of these
        Parts to be processed
    """
    for part in iter_parts(parts):
        # page, system, loudnessdirection, tempodirection
        _set_end_times(part, Page)
        _set_end_times(part, System)
        _set_end_times(part, ConstantLoudnessDirection)
        _set_end_times(part, ConstantTempoDirection)
        _set_end_times(part, ConstantArticulationDirection)
    

def _set_end_times(part, cls):
    acc = []
    t = None

    for obj in part.iter_all(cls, include_subclasses=True):

        if obj.start == t:

            if obj.end is None:

                acc.append(obj)

        else:

            for o in acc:

                part.add(o, end=obj.start.t)

            acc = []

            if obj.end is None:

                acc.append(obj)

            t = obj.start

    for o in acc:

        part.add(o, end=part.last_point.t)


def split_note(part, note, splits):
    # non-public

    # TODO: we shouldn't do this, but for now it's a good sanity check
    assert len(splits) > 0
    # TODO: we shouldn't do this, but for now it's a good sanity check
    assert note.symbolic_duration is None
    part.remove(note)
    divs_map = part.quarter_duration_map
    orig_tie_next = note.tie_next
    slur_stops = note.slur_stops
    cur_note = note
    start, end, sym_dur = splits.pop(0)
    cur_note.symbolic_duration = sym_dur
    part.add(cur_note, start, end)

    while splits:
        note.slur_stops = []

        if cur_note.id is not None:
            note_id = _make_tied_note_id(cur_note.id)
        else:
            note_id = None

        next_note = Note(note.step, note.octave, note.alter, voice=note.voice,
                         id=note_id, staff=note.staff)
        cur_note.tie_next = next_note
        next_note.tie_prev = cur_note

        cur_note = next_note
        start, end, sym_dur = splits.pop(0)
        cur_note.symbolic_duration = sym_dur

        part.add(cur_note, start, end)

    cur_note.tie_next = orig_tie_next

    if cur_note != note:
        # cur_note.slur_stops = slur_stops
        for slur in slur_stops:
            slur.end_note = cur_note


def find_tuplets(part):
    """TODO: Description

    Parameters
    ----------
    part: type
        Description of `part`

    """

    # quick shot at finding tuplets intended to cover some common cases.

    # are tuplets always in the same voice?

    # quite arbitrary:
    search_for_tuplets = [9, 7, 5, 3]
    # only look for x:2 tuplets
    normal_notes = 2

    candidates = []
    prev_end = None

    # 1. group consecutive notes without symbolic_duration
    for note in part.iter_all(GenericNote, include_subclasses=True):

        if note.symbolic_duration is None:
            if note.start.t == prev_end:
                candidates[-1].append(note)
            else:
                candidates.append([note])
            prev_end = note.end.t

    # 2. within each group
    for group in candidates:

        # 3. search for the predefined list of tuplets
        for actual_notes in search_for_tuplets:

            if actual_notes > len(group):
                # tuplet requires more notes than we have
                continue

            tup_start = 0

            while tup_start <= (len(group) - actual_notes):
                note_tuplet = group[tup_start:tup_start + actual_notes]
                # durs = set(n.duration for n in group[:tuplet-1])
                durs = set(n.duration for n in note_tuplet)

                if len(durs) > 1:
                    # notes have different durations (possibly valid but not
                    # supported here)
                    # continue
                    tup_start += 1
                else:

                    start = note_tuplet[0].start.t
                    end = note_tuplet[-1].end.t
                    total_dur = end - start

                    # total duration of tuplet notes must be integer-divisble by
                    # normal_notes
                    if total_dur % normal_notes > 0:
                        # continue
                        tup_start += 1
                    else:
                        # estimate duration type
                        dur_type = estimate_symbolic_duration(total_dur // normal_notes,
                                                              note_tuplet[0].start.quarter)
                        # int(divs_map(start)))

                        if dur_type and dur_type.get('dots', 0) == 0:
                            # recognized duration without dots
                            dur_type['actual_notes'] = actual_notes
                            dur_type['normal_notes'] = normal_notes
                            for note in note_tuplet:
                                note.symbolic_duration = dur_type.copy()
                            start_note = note_tuplet[0]
                            stop_note = note_tuplet[-1]
                            tuplet = Tuplet(start_note, stop_note)
                            part.add(tuplet, start_note.start.t, stop_note.end.t)
                            tup_start += actual_notes

                        else:
                            tup_start += 1


class InvalidTimePointException():
    """Raised when a time point is instantiated with an invalid number.

    """

if __name__ == '__main__':
    import doctest
    doctest.testmod()
