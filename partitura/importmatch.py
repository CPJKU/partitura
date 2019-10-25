#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing matchfiles
"""
import re
from fractions import Fraction
from collections import defaultdict
from operator import attrgetter, itemgetter
import logging
import warnings

import numpy as np
from scipy.interpolate import interp1d

from partitura.utils import (
    pitch_spelling_to_midi_pitch,
    ensure_pitch_spelling_format,
    ALTER_SIGNS,
    key_name_to_fifths_mode,
    iter_current_next,
    partition,
    estimate_clef_properties,
    notes_to_notearray
)

from partitura.performance import PerformedPart

import partitura.score as score
from partitura.musicanalysis import estimate_voices, estimate_key

__all__ = ['load_match']
LOGGER = logging.getLogger(__name__)

rational_pattern = re.compile('^([0-9]+)/([0-9]+)$')
double_rational_pattern = re.compile('^([0-9]+)/([0-9]+)/([0-9]+)$')
LATEST_VERSION = 5.0

PITCH_CLASSES = [('C', 0), ('C', 1), ('D', 0), ('D', 1), ('E', 0), ('F', 0),
                 ('F', 1), ('G', 0), ('G', 1), ('A', 0), ('A', 1), ('B', 0)]

PC_DICT = dict(zip(range(12), PITCH_CLASSES))
CLEF_ORDER = ['G', 'F', 'C', 'percussion', 'TAB', 'jianpu', None]
NUMBER_PAT = re.compile('\d+')


class MatchError(Exception):
    pass


class FractionalSymbolicDuration(object):
    """
    A class to represent symbolic duration information
    """

    def __init__(self, numerator, denominator=1,
                 tuple_div=None, add_components=None):

        self.numerator = numerator
        self.denominator = denominator
        self.tuple_div = tuple_div
        self.add_components = add_components

    def _str(self, numerator, denominator, tuple_div):
        if denominator == 1 and tuple_div is None:
            return str(numerator)
        else:
            if tuple_div is None:
                return '{0}/{1}'.format(numerator,
                                        denominator)
            else:
                return '{0}/{1}/{2}'.format(numerator,
                                            denominator,
                                            tuple_div)

    def __str__(self):

        if self.add_components is None:
            return self._str(self.numerator,
                             self.denominator,
                             self.tuple_div)
        else:
            r = [self._str(*i) for i in self.add_components]
            return '+'.join(r)

    def __add__(self, sd):
        if isinstance(sd, int):
            sd = FractionalSymbolicDuration(sd, 1)

        dens = np.array([self.denominator, sd.denominator], dtype=np.int)
        new_den = np.lcm(dens[0], dens[1])
        a_mult = new_den // dens
        new_num = np.dot(a_mult, [self.numerator, sd.numerator])

        if (self.add_components is None
                and sd.add_components is None):
            add_components = [
                (self.numerator, self.denominator, self.tuple_div),
                (sd.numerator, sd.denominator, sd.tuple_div)]

        elif (self.add_components is not None and
              sd.add_components is None):
            add_components = (self.add_components +
                              [(sd.numerator, sd.denominator, sd.tuple_div)])
        elif (self.add_components is None and
              sd.add_components is not None):
            add_components = ([(self.numerator, self.denominator, self.tuple_div)] +
                              sd.add_components)
        else:
            add_components = self.add_components + sd.add_components

        # Remove spurious components with 0 in the numerator
        add_components = [c for c in add_components if c[0] != 0]

        return FractionalSymbolicDuration(
            numerator=new_num,
            denominator=new_den,
            add_components=add_components)

    def __radd__(self, sd):
        return self.__add__(sd)

    def __float__(self):
        return self.numerator / (self.denominator * (self.tuple_div or 1))


def interpret_field(data):
    """
    Convert data to int, if not possible, to float, otherwise return
    data itself.

    Parameters
    ----------
    data : object
       Some data object

    Returns
    -------
    data : int, float or same data type as the input
       Return the data object casted as an int, float or return
       data itself.
    """

    try:
        return int(data)
    except ValueError:
        try:
            return float(data)
        except ValueError:
            return data


def interpret_field_rational(data, allow_additions=True, rationals_as_list=True):
    """Convert data to int, if not possible, to float, if not possible
    try to interpret as rational number and return it as float, if not
    possible, return data itself."""
    # global rational_pattern
    v = interpret_field(data)
    if type(v) == str:
        m = rational_pattern.match(v)
        m2 = double_rational_pattern.match(v)
        if m:
            groups = m.groups()
            if rationals_as_list:
                return [int(g) for g in groups]
            else:
                return FractionalSymbolicDuration(*[int(g) for g in groups])
        elif m2:
            groups = m2.groups()
            if rationals_as_list:
                return [int(g) for g in groups]
            else:
                return FractionalSymbolicDuration(*[int(g) for g in groups])
        else:
            if allow_additions:
                parts = v.split('+')

                if len(parts) > 1:
                    iparts = [interpret_field_rational(
                        i, allow_additions=False, rationals_as_list=False) for i in parts]

                    # to be replaced with isinstance(i,numbers.Number)
                    if all(type(i) in (int, float, FractionalSymbolicDuration) for i in iparts):
                        if any([isinstance(i, FractionalSymbolicDuration) for i in iparts]):
                            iparts = [FractionalSymbolicDuration(i) if not isinstance(i, FractionalSymbolicDuration) else i
                                      for i in iparts]
                        return sum(iparts)
                    else:
                        return v
                else:
                    return v
            else:
                return v
    else:
        return v

###################################################


class MatchLine(object):

    out_pattern = ''
    field_names = []
    re_obj = re.compile('')
    field_interpreter = interpret_field_rational

    def __str__(self):
        r = [self.__class__.__name__]
        for fn in self.field_names:
            r.append(' {0}: {1}'.format(fn, self.__dict__[fn]))
        return '\n'.join(r) + '\n'

    @property
    def matchline(self):
        raise NotImplementedError

    @classmethod
    def match_pattern(cls, s, pos=0):
        return cls.re_obj.search(s, pos=pos)

    @classmethod
    def from_matchline(cls, matchline, pos=0):
        match_pattern = cls.re_obj.search(matchline, pos)

        if match_pattern is not None:

            groups = [cls.field_interpreter(i) for i in match_pattern.groups()]
            kwargs = dict(zip(cls.field_names, groups))

            match_line = cls(**kwargs)

            return match_line

        else:
            raise MatchError('Input match line does not fit the expected pattern.')


class MatchInfo(MatchLine):
    out_pattern = 'info({Attribute},{Value}).'
    field_names = ['Attribute', 'Value']
    pattern = 'info\(\s*([^,]+)\s*,\s*(.+)\s*\)\.'
    re_obj = re.compile(pattern)
    field_interpreter = interpret_field

    def __init__(self, Attribute, Value):
        self.Attribute = Attribute
        self.Value = Value

    @property
    def matchline(self):
        return self.out_pattern.format(
            Attribute=self.Attribute,
            Value=self.Value)


class MatchMeta(MatchLine):

    out_pattern = 'meta({Attribute},{Value},{Bar},{TimeInBeats}).'
    field_names = ['Attribute', 'Value', 'Bar', 'TimeInBeats']
    pattern = 'meta\(\s*([^,]*)\s*,\s*([^,]*)\s*,\s*([^,]*)\s*,\s*([^,]*)\s*\)\.'
    re_obj = re.compile(pattern)
    field_interpreter = interpret_field

    def __init__(self, Attribute, Value, Bar, TimeInBeats):
        self.Attribute = Attribute
        self.Value = Value
        self.Bar = Bar
        self.TimeInBeats = TimeInBeats

    @property
    def matchline(self):
        return self.out_pattern.format(
            Attribute=self.Attribute,
            Value=self.Value,
            Bar=self.Bar,
            TimeInBeats=self.TimeInBeats)


class MatchSnote(MatchLine):
    """
    Class representing a score note
    """

    out_pattern = ('snote({Anchor},[{NoteName},{Modifier}],{Octave},'
                   + '{Bar}:{Beat},{Offset},{Duration},'
                   + '{OnsetInBeats},{OffsetInBeats},'
                   + '[{ScoreAttributesList}])')

    pattern = 'snote\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+):([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),\[(.*)\]\)'
    re_obj = re.compile(pattern)

    field_names = ['Anchor', 'NoteName', 'Modifier', 'Octave',
                   'Bar', 'Beat', 'Offset', 'Duration',
                   'OnsetInBeats', 'OffsetInBeats', 'ScoreAttributesList']

    def __init__(self, Anchor, NoteName, Modifier, Octave, Bar, Beat,
                 Offset, Duration, OnsetInBeats, OffsetInBeats,
                 ScoreAttributesList=[]):

        self.Anchor = str(Anchor)
        (self.NoteName, self.Modifier,
         self.Octave) = ensure_pitch_spelling_format(
             step=NoteName,
            alter=Modifier,
            octave=Octave)

        self.Bar = Bar
        self.Beat = Beat

        if isinstance(Offset, int):
            self.Offset = FractionalSymbolicDuration(Offset, 1)
        elif isinstance(Offset, (list, tuple)):
            self.Offset = FractionalSymbolicDuration(*Offset)
        elif isinstance(Offset, FractionalSymbolicDuration):
            self.Offset = Offset

        if isinstance(Duration, int):
            self.Duration = FractionalSymbolicDuration(Duration, 1)
        elif isinstance(Duration, (list, tuple)):
            self.Duration = FractionalSymbolicDuration(*Duration)
        elif isinstance(Duration, FractionalSymbolicDuration):
            self.Duration = Duration

        self.OnsetInBeats = OnsetInBeats
        self.OffsetInBeats = OffsetInBeats

        if isinstance(ScoreAttributesList, (list, tuple, np.ndarray)):
            # Always cast ScoreAttributesList as list?
            self.ScoreAttributesList = list(ScoreAttributesList)
        elif isinstance(ScoreAttributesList, str):
            self.ScoreAttributesList = ScoreAttributesList.split(',')
        elif isinstance(ScoreAttributesList, (int, float)):
            self.ScoreAttributesList = [ScoreAttributesList]
        else:
            raise ValueError('`ScoreAttributesList` must be a list or a string')

        # Cast all attributes in ScoreAttributesList as strings
        self.ScoreAttributesList = [str(sa) for sa in self.ScoreAttributesList]

    @property
    def DurationInBeats(self):
        return self.OffsetInBeats - self.OnsetInBeats

    @property
    def DurationSymbolic(self):
        if isinstance(self.Duration, FractionalSymbolicDuration):
            return str(self.Duration)
        elif isinstance(self.Duration, (float, int)):
            return str(Fraction.from_float(self.Duration))
        elif isinstance(self.Duration, str):
            return self.Duration

    @property
    def MidiPitch(self):
        if isinstance(self.Octave, int):
            return pitch_spelling_to_midi_pitch(step=self.NoteName,
                                                octave=self.Octave,
                                                alter=self.Modifier)
        else:
            return None

    @property
    def matchline(self):
        return self.out_pattern.format(
            Anchor=self.Anchor,
            NoteName=self.NoteName,
            Modifier='n' if self.Modifier == 0 else ALTER_SIGNS[self.Modifier],
            Octave=self.Octave,
            Bar=self.Bar,
            Beat=self.Beat,
            Offset=str(self.Offset),
            Duration=self.DurationSymbolic,
            OnsetInBeats=self.OnsetInBeats,
            OffsetInBeats=self.OffsetInBeats,
            ScoreAttributesList=','.join(self.ScoreAttributesList))


class MatchNote(MatchLine):
    """
    Class representing the performed note part of a match line
    """
    out_pattern = ('note({Number},[{NoteName},{Modifier}],' +
                   '{Octave},{Onset},{Offset},{AdjOffset},{Velocity})')

    field_names = ['Number', 'NoteName', 'Modifier', 'Octave',
                   'Onset', 'Offset', 'AdjOffset', 'Velocity']
    pattern = 'note\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)\)'

    re_obj = re.compile(pattern)

    # For backwards compatibility with Matchfile Version 1
    field_names_v1 = ['Number', 'NoteName', 'Modifier', 'Octave',
                      'Onset', 'Offset', 'Velocity']
    pattern_v1 = 'note\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+),([^,]+),([^,]+)\)'
    re_obj_v1 = re.compile(pattern_v1)

    def __init__(self, Number, NoteName, Modifier,
                 Octave, Onset, Offset, AdjOffset,
                 Velocity, MidiPitch=None, version=LATEST_VERSION):

        self.Number = Number

        # check if all pitch spelling information was provided
        has_pitch_spelling = not (
            NoteName is None or Modifier is None or Octave is None)

        # check if the MIDI pitch of the note was provided
        has_midi_pitch = MidiPitch is not None

        # Raise an error if neither pitch spelling nor MIDI pitch were provided
        if not has_pitch_spelling and not has_midi_pitch:
            raise ValueError('No note height information provided!')

        # Set attributes regarding pitch spelling
        if has_pitch_spelling:
            # Ensure that note name is uppercase
            (self.NoteName, self.Modifier,
             self.Octave) = ensure_pitch_spelling_format(
                 step=NoteName,
                 alter=Modifier,
                 octave=Octave)

        else:
            # infer the pitch information from the MIDI pitch
            # Note that this is just a dummy method, and does not correspond to
            # musically correct pitch spelling.
            self.NoteName, self.Modifier = PC_DICT[int(np.mod(MidiPitch, 12))]
            self.Octave = int(MidiPitch // 12 - 1)

        # Check if the provided MIDI pitch corresponds to the correct pitch spelling
        if has_midi_pitch:
            if MidiPitch != pitch_spelling_to_midi_pitch(
                    step=self.NoteName,
                    octave=self.Octave,
                    alter=self.Modifier):
                raise ValueError('The provided pitch spelling information does not match '
                                 'the given MIDI pitch!')

            else:
                # Set the Midi pitch
                self.MidiPitch = int(MidiPitch)

        else:
            self.MidiPitch = pitch_spelling_to_midi_pitch(
                step=self.NoteName,
                octave=self.Octave,
                alter=self.Modifier)

        self.Onset = Onset
        self.Offset = Offset
        self.AdjOffset = AdjOffset

        if AdjOffset is None:
            # Raise warning!
            self.AdjOffset = self.Offset

        self.Velocity = int(Velocity)

        # TODO
        # * check version and update necessary patterns
        self.version = version

    @property
    def matchline(self):
        return self.out_pattern.format(
            Number=self.Number,
            NoteName=self.NoteName,
            Modifier='n' if self.Modifier == 0 else ALTER_SIGNS[self.Modifier],
            Octave=self.Octave,
            Onset=self.Onset,
            Offset=self.Offset,
            AdjOffset=self.AdjOffset,
            Velocity=self.Velocity)

    @property
    def Duration(self):
        return self.Offset - self.Onset

    def AdjDuration(self):
        return self.AdjOffset - self.Onset

    @classmethod
    def from_matchline(cls, matchline, pos=0):
        """Create a MatchNote from a line

        """
        match_pattern = cls.re_obj.search(matchline, pos)

        if match_pattern is None:
            match_pattern = cls.re_obj_v1.search(matchline, pos)

            if match_pattern is not None:
                groups = [cls.field_interpreter(i) for i in match_pattern.groups()]
                kwargs = dict(zip(cls.field_names_v1, groups))
                kwargs['version'] = 1.0
                kwargs['AdjOffset'] = None
                match_line = cls(**kwargs)

                return match_line
            else:
                raise MatchError('Input matchline does not fit expected pattern')

        else:
            groups = [cls.field_interpreter(i) for i in match_pattern.groups()]
            kwargs = dict(zip(cls.field_names, groups))
            match_line = cls(**kwargs)
            return match_line


class MatchSnoteNote(MatchLine):
    """
    Class representing a "match" (containing snote and note)

    TODO:
    * More readable __str__ method

    """

    out_pattern = '{SnoteLine}-{NoteLine}.'
    pattern = MatchSnote.pattern + '-' + MatchNote.pattern
    re_obj = re.compile(pattern)
    field_names = MatchSnote.field_names + MatchNote.field_names

    # for version 1
    pattern_v1 = MatchNote.pattern + '-' + MatchNote.pattern_v1
    re_obj_v1 = re.compile(pattern_v1)
    field_names_v1 = MatchSnote.field_names + MatchNote.field_names_v1

    def __init__(self, snote, note, same_pitch_spelling=True):
        self.snote = snote
        self.note = note

        # Set the same pitch spelling in both note and snote
        # (this can break if the snote is not exactly matched
        # to a note with the same pitch). Handle with care.
        if same_pitch_spelling:
            self.note.NoteName = self.snote.NoteName
            self.note.Modifier = self.snote.Modifier
            self.note.Octave = self.snote.Octave
            self.note.MidiPitch = self.snote.MidiPitch

    @property
    def matchline(self):
        return self.out_pattern.format(
            SnoteLine=self.snote.matchline,
            NoteLine=self.note.matchline)

    @classmethod
    def from_matchline(cls, matchline, pos=0):
        match_pattern = cls.re_obj.search(matchline, pos=0)

        if match_pattern is None:
            match_pattern = cls.re_obj_v1.search(matchline, pos)

            if match_pattern is not None:
                groups = [cls.field_interpreter(i) for i in match_pattern.groups()]

                snote_kwargs = dict(zip(MatchSnote.field_names,
                                        groups[:len(MatchSnote.field_names)]))
                note_kwargs = dict(zip(MatchNote.field_names_v1,
                                       groups[len(MatchSnote.field_names):]))
                note_kwargs['version'] = 1.0
                note_kwargs['AdjOffset'] = None
                snote = MatchSnote(**snote_kwargs)
                note = MatchNote(**note_kwargs)
                match_line = cls(snote=snote,
                                 note=note)

                return match_line
            else:
                raise MatchError('Input matchline does not fit expected pattern')

        else:
            groups = [cls.field_interpreter(i) for i in match_pattern.groups()]
            snote_kwargs = dict(zip(MatchSnote.field_names,
                                    groups[:len(MatchSnote.field_names)]))
            note_kwargs = dict(zip(MatchNote.field_names,
                                   groups[len(MatchSnote.field_names):]))
            snote = MatchSnote(**snote_kwargs)
            note = MatchNote(**note_kwargs)
            match_line = cls(snote=snote,
                             note=note)
            return match_line

    def __str__(self):
        # TODO:
        # Nicer print?
        return str(self.snote) + '\n' + str(self.note)


class MatchSnoteDeletion(MatchLine):
    """
    Class for representing a deleted note, i.e., a score note
    which was not performed.
    """
    out_pattern = '{SnoteLine}-deletion.'
    pattern = MatchSnote.pattern + '-deletion\.'
    re_obj = re.compile(pattern)
    field_names = MatchSnote.field_names

    def __init__(self, snote):
        self.snote = snote

    @property
    def matchline(self):
        return self.out_pattern.format(
            SnoteLine=self.snote.matchline)

    @classmethod
    def from_matchline(cls, matchline, pos=0):
        match_pattern = cls.re_obj.search(matchline, pos=0)

        if match_pattern is not None:
            groups = [cls.field_interpreter(i) for i in match_pattern.groups()]
            snote_kwargs = dict(zip(MatchSnote.field_names, groups))
            snote = MatchSnote(**snote_kwargs)
            match_line = cls(snote=snote)
            return match_line

        else:
            raise MatchError('Input matchline does not fit expected pattern')

    def __str__(self):
        return str(self.snote) + '\nDeletion'


class MatchSnoteTrailingScore(MatchSnoteDeletion):
    """
    A variant of MatchSnoteDeletion for older match files.
    """
    pattern = MatchSnote.pattern + '-trailing_score_note\.'
    re_obj = re.compile(pattern)

    def __init__(self, snote):
        super().__init__(snote)


class MatchInsertionNote(MatchLine):
    """
    A class for representing inserted notes, i.e., performed notes
    without a corresponding score note.
    """
    out_pattern = 'insertion-{NoteLine}.'
    pattern = 'insertion-' + MatchNote.pattern + '.'
    re_obj = re.compile(pattern)
    field_names = MatchNote.field_names

    def __init__(self, note):
        self.note = note
        for fn in self.field_names:
            setattr(self, fn, getattr(self.note, fn, None))

    @property
    def matchline(self):
        return self.out_pattern.format(
            NoteLine=self.note.matchline)

    @classmethod
    def from_matchline(cls, matchline, pos=0):
        match_pattern = cls.match_pattern(matchline, pos=0)

        if match_pattern is not None:
            groups = [cls.field_interpreter(i) for i in match_pattern.groups()]
            note_kwargs = dict(zip(MatchNote.field_names, groups))
            note = MatchNote(**note_kwargs)
            return cls(note=note)
        else:
            raise MatchError('Input matchline does not fit expected pattern')


class MatchHammerBounceNote(MatchInsertionNote):
    """
    A variant of MatchInsertionNote for older match files.
    """
    pattern = 'hammer_bounce-' + MatchNote.pattern + '.'
    re_obj = re.compile(pattern)

    def __init__(self, note):
        super().__init__(note)


class MatchTrailingPlayedNote(MatchInsertionNote):
    """
    Another variant of MatchInsertionNote for older match files.
    """
    pattern = 'trailing_played_note-' + MatchNote.pattern + '.'
    re_obj = re.compile(pattern)

    def __init__(self, note):
        super().__init__(note)


class MatchSustainPedal(MatchLine):
    """
    Class for representing a sustain pedal line
    """
    out_pattern = 'sustain({Time},{Value}).'
    field_names = ['Time', 'Value']
    pattern = 'sustain\(\s*([^,]*)\s*,\s*([^,]*)\s*\)\.'
    re_obj = re.compile(pattern)

    def __init__(self, Time, Value):
        self.Time = Time
        self.Value = Value

    @property
    def matchline(self):

        return self.out_pattern.format(
            Time=self.Time,
            Value=self.Value)


class MatchSoftPedal(MatchLine):
    """
    Class for representing a soft pedal line
    """
    out_pattern = 'soft({Time},{Value}).'
    field_names = ['Time', 'Value']
    pattern = 'soft\(\s*([^,]*)\s*,\s*([^,]*)\s*\)\.'
    re_obj = re.compile(pattern)

    def __init__(self, Time, Value):
        self.Time = Time
        self.Value = Value

    @property
    def matchline(self):

        return self.out_pattern.format(
            Time=self.Time,
            Value=self.Value)


class MatchOrnamentNote(MatchLine):
    out_pattern = 'ornament({Anchor})-{NoteLine}'
    field_names = ['Anchor'] + MatchNote.field_names
    pattern = 'ornament\(([^\)]*)\)-' + MatchNote.pattern
    re_obj = re.compile(pattern)

    def __init__(self, Anchor, note):
        self.Anchor = Anchor
        self.note = note

    @property
    def matchline(self):
        return self.out_pattern.format(
            Anchor=self.Anchor,
            NoteLine=self.note.matchline)

    @classmethod
    def from_matchline(cls, matchline, pos=0):
        match_pattern = cls.match_pattern(matchline, pos=0)

        if match_pattern is not None:
            groups = [cls.field_interpreter(i) for i in match_pattern.groups()]
            note_kwargs = dict(zip(MatchNote.field_names, groups[1:]))

            anchor = groups[0]
            note = MatchNote(**note_kwargs)
            return cls(Anchor=anchor,
                       note=note)

        else:
            raise MatchError('Input matchline does not fit expected pattern')


class MatchTrillNote(MatchOrnamentNote):
    out_pattern = 'trill({Anchor})-{NoteLine}'
    field_names = ['Anchor'] + MatchNote.field_names
    pattern = 'trill\(([^\)]*)\)-' + MatchNote.pattern
    re_obj = re.compile(pattern)

    def __init__(self, Anchor, note):
        super().__init__(Anchor, note)


def parse_matchline(l):
    """
    Return objects representing the line as one of:

    * hammer_bounce-PlayedNote.
    * info(Attribute, Value).
    * insertion-PlayedNote.
    * ornament(Anchor)-PlayedNote.
    * ScoreNote-deletion.
    * ScoreNote-PlayedNote.
    * ScoreNote-trailing_score_note.
    * trailing_played_note-PlayedNote.
    * trill(Anchor)-PlayedNote.
    * meta(Attribute, Value, Bar, Beat).
    * sustain(Time, Value)
    * soft(Time, Value)

    or False if none can be matched

    Parameters
    ----------
    l : str
        Line of the match file

    Returns
    -------
    matchline : subclass of `MatchLine`
       Object representing the line.
    """

    from_matchline_methods = [MatchSnoteNote.from_matchline,
                              MatchSnoteDeletion.from_matchline,
                              MatchSnoteTrailingScore.from_matchline,
                              MatchInsertionNote.from_matchline,
                              MatchHammerBounceNote.from_matchline,
                              MatchTrailingPlayedNote.from_matchline,
                              MatchTrillNote.from_matchline,
                              MatchOrnamentNote.from_matchline,
                              MatchSustainPedal.from_matchline,
                              MatchSoftPedal.from_matchline,
                              MatchInfo.from_matchline,
                              MatchMeta.from_matchline,
                              ]
    matchline = False
    for from_matchline in from_matchline_methods:
        try:
            matchline = from_matchline(l)
            break
        except MatchError:
            continue

    return matchline


class MatchFile(object):
    """
    Class for representing MatchFiles
    """

    def __init__(self, filename):

        self.name = filename

        with open(filename) as f:

            self.lines = np.array([parse_matchline(l) for l in f.read().splitlines()])

    @property
    def note_pairs(self):
        """
        Return all(snote, note) tuples

        """
        return [(x.snote, x.note) for x in self.lines if isinstance(x, MatchSnoteNote)]

    @property
    def notes(self):
        """
        Return all performed notes (as MatchNote objects)
        """
        return [x.note for x in self.lines if hasattr(x, 'note')]

    def iter_notes(self):
        """
        Iterate over all performed notes (as MatchNote objects)
        """
        for x in self.lines:
            if hasattr(x, 'note'):
                yield x.note

    @property
    def snotes(self):
        """
        Return all score notes (as MatchSnote objects)
        """
        return [x.snote for x in self.lines if hasattr(x, 'snote')]

    def iter_snotes(self):
        """
        Iterate over all performed notes (as MatchNote objects)
        """
        for x in self.lines:
            if hasattr(x, 'snote'):
                yield x.snote

    @property
    def sustain_pedal(self):
        return [l for l in self.lines if isinstance(l, MatchSustainPedal)]

    @property
    def insertions(self):
        return [x.note for x in self.lines if isinstance(x, MatchInsertionNote)]

    @property
    def deletions(self):
        return [x.snote for x in self.lines if isinstance(x, MatchSnoteDeletion)]

    @property
    def _info(self):
        """
        Return all InfoLine objects

        """
        return [i for i in self.lines if isinstance(i, MatchInfo)]

    def info(self, attribute=None):
        """
        Return the value of the MatchInfo object corresponding to
        attribute, or None if there is no such object

        : param attribute: the name of the attribute to return the value for

        """
        if attribute:
            try:
                idx = [i.Attribute for i in self._info].index(attribute)
                return self._info[idx].Value
            except:
                return None
        else:
            return self._info

    @property
    def first_onset(self):
        return min([n.OnsetInBeats for n in self.snotes])

    @property
    def first_bar(self):
        return min([n.Bar for n in self.snotes])

    @property
    def time_signatures(self):
        """
        A list of tuples(t, b, (n, d)), indicating a time signature of n over v, starting at t in bar b

        """
        tspat = re.compile('([0-9]+)/([0-9]*)')
        m = [(int(x[0]), int(x[1])) for x in
             tspat.findall(self.info('timeSignature'))]
        _timeSigs = []
        if len(m) > 0:
            _timeSigs.append((self.first_onset, self.first_bar, m[0]))
        for l in self.time_sig_lines:
            _timeSigs.append((float(l.TimeInBeats), int(l.Bar), [
                            (int(x[0]), int(x[1])) for x in tspat.findall(l.Value)][0]))
        _timeSigs = list(set(_timeSigs))
        _timeSigs.sort(key=lambda x: x[0])

        # ensure that all consecutive time signatures are different
        timeSigs = [_timeSigs[0]]

        for ts in _timeSigs:
            ts_on, bar, (ts_num, ts_den) = ts
            ts_on_prev, ts_bar_prev, (ts_num_prev, ts_den_prev) = timeSigs[-1]
            if ts_num != ts_num_prev or ts_den != ts_den_prev:
                timeSigs.append(ts)

        return timeSigs

    def _time_sig_lines(self):
        return [i for i in self.lines if
                isinstance(i, MatchMeta)
                and hasattr(i, 'Attribute')
                and i.Attribute == 'timeSignature']

    @property
    def time_sig_lines(self):
        ml = self._time_sig_lines()
        if len(ml) == 0:
            ts = self.info('timeSignature')
            ml = [parse_matchline(
                'meta(timeSignature,{0},{1},{2}).'.format(ts, self.first_bar, self.first_onset))]
        return ml

    @property
    def key_signatures(self):
        """
        A list of tuples (t, b, (f,m)) or (t, b, (f1, m1, f2, m2))
        """
        kspat = re.compile(('(?P<step1>[A-G])(?P<alter1>[#b]*) '
                            '(?P<mode1>[a-zA-z]+)(?P<slash>/*)'
                            '(?P<step2>[A-G]*)(?P<alter2>[#b]*)'
                            '(?P<space2> *)(?P<mode2>[a-zA-z]*)'))

        _keysigs = []
        for ksl in self.key_sig_lines:
            if isinstance(ksl, MatchInfo):
                # remove brackets and only take
                # the first key signature
                ks = ksl.Value.replace('[', '').replace(']', '').split(',')[0]
                t = self.first_onset
                b = self.first_bar
            elif isinstance(ksl, MatchMeta):
                ks = ksl.Value
                t = ksl.TimeInBeats
                b = ksl.Bar

            ks_info = kspat.search(ks)

            keysig = (self._format_key_signature(
                step=ks_info.group('step1'),
                alter_sign=ks_info.group('alter1'),
                mode=ks_info.group('mode1')), )

            if ks_info.group('step2') != '':
                keysig += (self._format_key_signature(
                    step=ks_info.group('step2'),
                    alter_sign=ks_info.group('alter2'),
                    mode=ks_info.group('mode2')), )

            _keysigs.append((t, b, keysig))

        keysigs = [_keysigs[0]]

        for k in _keysigs:
            if k[2] != keysigs[-1][2]:
                keysigs.append(k)

        return keysigs

    def _format_key_signature(self, step, alter_sign, mode):

        if mode.lower() in ('maj', '', 'major'):
            mode = ''
        elif mode.lower() in ('min', 'm', 'minor'):
            mode = 'm'
        else:
            raise ValueError('Invalid mode. Expected "major" or "minor" but got {0}'.format(mode))

        return step + alter_sign + mode

    @property
    def key_sig_lines(self):

        ks_info = [l for l in self.info() if l.Attribute == 'keySignature']
        ml = (ks_info +
              [i for i in self.lines if
               isinstance(i, MatchMeta) and
               hasattr(i, 'Attribute') and
               i.Attribute == 'keySignature'])

        return ml

    def write(self, filename):
        with open(filename, 'w') as f:
            for l in self.lines:
                f.write(l.matchline + '\n')


def load_match(fn, create_part=False, pedal_threshold=64, first_note_at_zero=False):
    """Load a matchfile.

    Parameters
    ----------
    fn : str
        The matchfile
    create_part : bool, optional
        When True create a Part object from the snote information in
        the match file. Defaults to False.
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.

    Returns
    -------
    ppart : list
        The performed part, a list of dictionaries
    alignment : list
        The score--performance alignment, a list of dictionaries
    spart : Part
        The score part. This item is only returned when `create_part` = True.
    
    """
    # Parse Matchfile
    mf = MatchFile(fn)

    ######## Generate PerformedPart #########
    ppart = performed_part_from_match(mf, pedal_threshold, first_note_at_zero)
    ####### Generate Part ######
    if create_part:
        spart = part_from_matchfile(mf)
    ###### Alignment ########
    alignment = alignment_from_matchfile(mf)
    
    if create_part:
        return ppart, alignment, spart
    else:
        return ppart, alignment


def alignment_from_matchfile(mf):
    result = []

    for l in mf.lines:

        if isinstance(l, MatchSnoteNote):
            result.append(dict(label='match',
                               score_id=l.snote.Anchor,
                               performance_id=l.note.Number))
        elif isinstance(l, MatchSnoteDeletion):
            result.append(dict(label='deletion',
                               score_id=l.snote.Anchor))
        elif isinstance(l, MatchInsertionNote):
            result.append(dict(label='insertion',
                               performance_id=l.note.Number))
        elif isinstance(l, MatchOrnamentNote):
            if isinstance(l, MatchTrillNote):
                ornament_type = 'trill'
            else:
                ornament_type = 'generic_ornament'
            result.append(dict(label='ornament',
                               score_id=l.Anchor,
                               performance_id=l.note.Number,
                               type=ornament_type))

    return result


# PART FROM MATCHFILE stuff

def sort_snotes(snotes):
    sidx = np.lexsort(list(zip(*[(float(n.Offset), float(n.Beat), float(n.Bar))
                                 for n in snotes])))
    return [snotes[i] for i in sidx if snotes[i].NoteName.lower() != 'r']


def part_from_matchfile(mf):
    part = score.Part('P1', mf.info('piece'))
    # snotes = sorted(mf.snotes, key=attrgetter('OnsetInBeats'))
    snotes = sort_snotes(mf.snotes)
    divs = np.lcm.reduce(np.unique([note.Offset.denominator * (note.Offset.tuple_div or 1)
                                    for note in snotes]))
    part.set_quarter_duration(0, divs)
    min_time = snotes[0].OnsetInBeats  # sorted by OnsetInBeats
    max_time = max(n.OffsetInBeats for n in snotes)

    ts = mf.time_signatures

    beats_map, beat_type_map, min_time_q, max_time_q = make_timesig_maps(ts, max_time)

    bars = np.unique([n.Bar for n in snotes])
    t = min_time
    t = t * 4 / beat_type_map(min_time_q)
    offset = t
    # bar map: bar_number-> start in quarters
    bar_times = {}
    for b0, b1 in iter_current_next(bars, start=0):

        bar_times.setdefault(b1, t)
        if t < 0:
            t = 0
        else:
            # multiply by diff between consecutive bar numbers
            n_bars = b1 - b0
            if t <= max_time_q:
                t += (n_bars * 4 * beats_map(t)) / beat_type_map(t)

    for note in snotes:
        # start of bar in quarter units
        bar_start = bar_times[note.Bar]

        # offset within bar in quarter units
        bar_offset = (note.Beat - 1) * 4 / beat_type_map(bar_start)
        # offset within beat in quarter units
        beat_offset = (4 * note.Offset.numerator
                       / (note.Offset.denominator * (note.Offset.tuple_div or 1)))

        # # anacrusis
        if bar_start < 0:
            # in case of anacrusis we set the bar_start to -bar_duration (in
            # quarters) so that the below calculation is correct
            bar_start = - beats_map(bar_start) * 4 / beat_type_map(bar_start)

        # note onset in divs
        onset_divs = int(divs * (bar_start + bar_offset + beat_offset - offset))
        # print(note.Anchor, onset_divs, bar_start, bar_offset, beat_offset, offset)

        articulations = set()
        if 'staccato' in note.ScoreAttributesList:
            articulations.add('staccato')
        if 'accent' in note.ScoreAttributesList:
            articulations.add('accent')

        # dictionary with keyword args with which the Note (or GraceNote) will be instantiated
        note_attributes = dict(step=note.NoteName,
                               octave=note.Octave,
                               alter=note.Modifier,
                               id=note.Anchor,
                               articulations=articulations)

        staff_nr = next((a[-1] for a in note.ScoreAttributesList if a.startswith('staff')), None)
        try:
            note_attributes['staff'] = int(staff_nr)
        except (TypeError, ValueError):
            # no staff attribute, or staff attribute does not end with a number
            note_attributes['staff'] = None

        note_attributes['voice'] = next((int(a) for a in note.ScoreAttributesList
                                         if NUMBER_PAT.match(a)), None)

        # get rid of this if as soon as we have a way to iterate over the
        # duration components. For now we have to treat the cases simple
        # and compound durations separately.

        if note.Duration.add_components:
            prev_part_note = None

            for i, (num, den, tuple_div) in enumerate(note.Duration.add_components):

                # when we add multiple notes that are tied, the first note will
                # get the original note id, and subsequent notes will get a
                # derived note id (by appending, 'a', 'b', 'c',...)
                if i > 0:
                    # tnote_id = 'n{}_{}'.format(note.Anchor, i)
                    note_attributes['id'] = score.make_tied_note_id(note_attributes['id'])

                part_note = score.Note(**note_attributes)

                duration_divs = int(divs * 4 * num / (den * (tuple_div or 1)))

                assert duration_divs > 0

                offset_divs = onset_divs + duration_divs

                part.add(part_note, onset_divs, offset_divs)

                if prev_part_note:
                    prev_part_note.tie_next = part_note
                    part_note.tie_prev = prev_part_note
                prev_part_note = part_note
                onset_divs = offset_divs

        else:

            num = note.Duration.numerator
            den = note.Duration.denominator
            tuple_div = note.Duration.tuple_div
            duration_divs = int(divs * 4 * num / (den * (tuple_div or 1)))

            offset_divs = onset_divs + duration_divs

            # notes with duration 0, are also treated as grace notes, even if
            # they do not have a 'grace' score attribute
            if ('grace' in note.ScoreAttributesList or
                    note.Duration.numerator == 0):

                part_note = score.GraceNote('appoggiatura', **note_attributes)

            else:

                part_note = score.Note(**note_attributes)

            part.add(part_note, onset_divs, offset_divs)

    # add time signatures
    for (ts_beat_time, ts_bar, (ts_beats, ts_beat_type)) in ts:

        bar_start_divs = int(divs * (bar_times[ts_bar] - offset))  # in quarters
        part.add(score.TimeSignature(ts_beats, ts_beat_type), bar_start_divs)

    # add key signatures
    for (ks_beat_time, ks_bar, keys) in mf.key_signatures:

        if len(keys) > 1:
            # there are multple equivalent keys, so we check which one is most
            # likely according to the key estimator
            est_keys = estimate_key(notes_to_notearray(part.notes_tied), return_sorted_keys=True)
            idx = [est_keys.index(key) if key in est_keys else np.inf
                   for key in keys]
            key_name = keys[np.argmin(idx)]

        else:

            key_name = keys[0]

        fifths, mode = key_name_to_fifths_mode(key_name)
        part.add(score.KeySignature(fifths, mode), 0)

    add_staffs(part)
    # add_clefs(part)

    # add incomplete measure if necessary

    if offset < 0:

        part.add(score.Measure(number=1), 0, int(-offset * divs))

    # add the rest of the measures automatically
    score.add_measures(part)
    # print(part.pretty())
    score.tie_notes(part)
    score.find_tuplets(part)

    if not all([n.voice for n in part.notes_tied]):
        # print('notes without voice detected')
        add_voices(part)

    return part


def make_timesig_maps(ts_orig, max_time):
    # TODO: make sure that ts_orig covers range from min_time

    # return two functions that map score times (in quarter units) to time sig
    # beats, and time sig beat_type respectively
    ts = list(ts_orig)
    assert len(ts) > 0
    ts.append((max_time, None, ts[-1][2]))

    x = np.array([t for t, _, _ in ts])
    y = np.array([x for _, _, x in ts])

    start_q = x[0] * 4 / y[0, 1]
    x_q = np.cumsum(np.r_[start_q, 4 * np.diff(x) / y[:-1, 1]])
    end_q = x_q[-1]

    beats_map = interp1d(x_q, y[:, 0], kind='previous')
    beat_type_map = interp1d(x_q, y[:, 1], kind='previous')

    return beats_map, beat_type_map, start_q, end_q


def add_staffs(part, split=55, only_missing=True):
    # assign staffs using a hard limit
    notes = part.notes_tied
    for n in notes:

        if only_missing and n.staff:
            continue

        if n.midi_pitch > split:
            staff = 1
        else:
            staff = 2

        n.staff = staff

        n_tied = n.tie_next
        while n_tied:
            n_tied.staff = staff
            n_tied = n_tied.tie_next

    part.add(score.Clef(number=1, sign='G', line=2, octave_change=0), 0)
    part.add(score.Clef(number=2, sign='F', line=4, octave_change=0), 0)


def add_staffs_v1(part):
    # assign staffs by first estimating voices jointly, then assigning voices to staffs

    notes = part.notes_tied
    # estimate voices in strictly monophonic way
    voices = estimate_voices(notes_to_notearray(notes), monophonic_voices=True)
    # for v, note in zip(voices, notes):
    #     print(note.start.t, note.midi_pitch, v)

    # group notes by voice
    by_voice = partition(itemgetter(0), zip(voices, notes))
    clefs = {}

    for v, vnotes in by_voice.items():
        # voice numbers may be recycled throughout the piece, so we split by
        # time gap
        t_diffs = np.diff([n.start.t for _, n in vnotes])
        t_threshold = np.inf  # np.median(t_diffs)+1
        note_groups = np.split([note for _, note in vnotes],
                               np.where(t_diffs > t_threshold)[0] + 1)

        # for each note group estimate the clef
        for note_group in note_groups:
            if len(note_group) > 0:
                pitches = [n.midi_pitch for n in note_group]
                clef = tuple(estimate_clef_properties(pitches).items())
                staff = clefs.setdefault(clef, len(clefs))

                # print((note_group[0].start.t, note_group[-1].end.t),
                #       (np.min(pitches), np.max(pitches), np.mean(pitches)), clef)
                for n in note_group:
                    n.staff = staff
                    n_tied = n.tie_next
                    while n_tied:
                        n_tied.staff = staff
                        n_tied = n_tied.tie_next

    # re-order the staffs to a fixed order (see CLEF_ORDER), rather than by
    # first appearance
    clef_list = list((dict(clef), i) for clef, i in clefs.items())
    clef_list.sort(key=lambda x: x[0].get('octave_change', 0))
    clef_list.sort(key=lambda x: CLEF_ORDER.index(x[0].get('sign', 'G')))
    staff_map = dict((j, i + 1) for i, (_, j) in enumerate(clef_list))
    for n in notes:
        n.staff = staff_map[n.staff]
    for i, (clef_properties, _) in enumerate(clef_list):
        part.add(score.Clef(number=i + 1, **clef_properties), 0)
    # partition(attrgetter('staff'), part.notes_tied)
    # **estimate_clef_properties([n.midi_pitch for n in notes])


def add_clefs(part):
    by_staff = partition(attrgetter('staff'), part.notes_tied)
    for staff, notes in by_staff.items():
        part.add(score.Clef(number=staff, **estimate_clef_properties([n.midi_pitch for n in notes])), 0)


def add_voices(part):
    by_staff = partition(attrgetter('staff'), part.notes_tied)
    max_voice = 0
    for staff, notes in by_staff.items():

        voices = estimate_voices(notes_to_notearray(notes))

        assert len(voices) == len(notes)
        for n, voice in zip(notes, voices):
            assert voice > 0
            n.voice = voice + max_voice

            n_next = n
            while n_next.tie_next:
                n_next = n_next.tie_next
                n_next.voice = voice + max_voice

        max_voice = np.max(voices)

    if any([n.voice is None for n in part.notes]):
        # Hack to add voices to notes not included in a staff!
        # not musically meaningful
        ev = 1
        for n in part.notes:
            if n.voice is None:
                n.voice = max_voice + ev
                ev += 1
        # import pdb
        # pdb.set_trace()


# PERFORMANCE PART FROM MATCHFILE stuff

def performed_part_from_match(mf, pedal_threshold=64, first_note_at_zero=False):
    """Make PerformedPart from performance info in a MatchFile

    Parameters
    ----------
    mf : MatchFile
        A MatchFile instance
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.

    Returns
    -------
    ppart : PerformedPart
        A performed part
    
    """
    # Get midi time units
    mpq = mf.info('midiClockRate')
    ppq = mf.info('midiClockUnits')

    # PerformedNote instances for all MatchNotes
    notes = []

    first_note = next(mf.iter_notes(), None)
    if first_note and first_note_at_zero:
        offset = first_note.Onset * mpq / (10**6 * ppq)
    else:
        offset = 0
        
    for note in mf.iter_notes():

        sound_off = note.Offset if note.AdjOffset is None else note.AdjOffset

        notes.append(dict(id=note.Number,
                          midi_pitch=note.MidiPitch,
                          note_on=note.Onset * mpq / (10**6 * ppq) - offset,
                          note_off=note.Offset * mpq / (10**6 * ppq) - offset,
                          sound_off=sound_off * mpq / (10**6 * ppq) - offset,
                          velocity=note.Velocity
                          ))

    # SustainPedal instances for sustain pedal lines
    sustain_pedal = []
    for ped in mf.sustain_pedal:
        sustain_pedal.append(dict(type='sustain_pedal',
                                  time=ped.Time * mpq / (10**6 * ppq),
                                  value=ped.Value))

    # Make performed part
    ppart = PerformedPart(id='P1',
                          part_name=mf.info('piece'),
                          notes=notes,
                          controls=sustain_pedal,
                          sustain_pedal_threshold=pedal_threshold)
    return ppart
