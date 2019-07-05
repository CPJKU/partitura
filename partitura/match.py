#!/usr/bin/env python

"""
This module is for accessing the data in match files accessible
from within python. It has only read capabilities, and is not
functionally complete in any way. But with the basics here provided
it should be relatively easy to extend the module to get the data
you want.
"""

import re
import numpy as np

rational_pattern = re.compile('^([0-9]+)/([0-9]+)$')
LATEST_VERSION = 4.0


def interpret_field(data):
    """
    Convert data to int, if not possible, to float, otherwise return
    data itself.

    :param data: some data object

    :returns:

    """

    try:
        return int(data)
    except ValueError:
        try:
            return float(data)
        except ValueError:
            return data


class ParseRationalException(Exception):
    def __init__(self, string):
        self.string = string

    def __str__(self):
        return 'Could not parse string "{0}"'.format(self.string)


class Ratio:
    def __init__(self, string):
        try:
            self.numerator, self.denominator = [
                int(i) for i in string.split('/')]
        except:
            raise ParseRationalException(string)


def interpret_field_rational(data, allow_additions=False):
    """Convert data to int, if not possible, to float, if not possible
    try to interpret as rational number and return it as float, if not
    possible, return data itself."""
    global rational_pattern
    v = interpret_field(data)
    if type(v) == str:
        m = rational_pattern.match(v)
        if m:
            groups = m.groups()
            return float(groups[0]) / float(groups[1])
        else:
            if allow_additions:
                parts = v.split('+')
                if len(parts) > 1:
                    iparts = [interpret_field_rational(
                        i, allow_additions=False) for i in parts]
                    # to be replaced with isinstance(i,numbers.Number)
                    if all(type(i) in (int, float) for i in iparts):
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


def pitch_name_2_midi_PC(modifier, name, octave):
    if name == 'r':
        return (0, 0)
    base_class = ({'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}[name.lower()]
                  + {'b': -1, 'bb': -2, '#': 1, 'x': 2, '##': 2, 'n': 0}[modifier])
    mid = (octave + 1) * 12 + base_class
    # for mozartmatch files (in which the octave numbers are off by one)
    # mid = octave*12 + base_class
    pitchclass = base_class % 12
    return (mid, pitchclass)


class MatchLine(object):
    """
    A class that represents a line in a match file. It is intended
    to be subclassed. It's constructor sets up a list of field names
    as object attributes.

    """
    field_names = []
    re_obj = re.compile('')

    def __str__(self):
        r = [self.__class__.__name__]
        for fn in self.field_names:
            r.append(' {0}: {1}'.format(fn, self.__dict__[fn]))
        return '\n'.join(r) + '\n'

    def __init__(self, match_obj, field_interpreter=interpret_field_rational):
        self.set_attributes(match_obj, field_interpreter)

    @classmethod
    def match_pattern(self, s, pos=0):
        """
        Return a regular expression match object that matches the
        pattern to the given string, or None if no match was found

        :param s: the string to be matched

        :returns: match object, or None

        """

        return self.re_obj.search(s, pos=pos)

    def set_attributes(self, match_obj, field_interpreter=lambda x: x):
        """
        Set attribute objects using values from a regular expression
        match object; use `field_interpreter` to interpret the
        attribute value strings as integers, floats, strings, etc.

        :param match_obj: regular expression match object
        :param field_interpreter: function that returns an object given a string

        """
        # import pdb
        # pdb.set_trace()
        groups = [field_interpreter(i) for i in match_obj.groups()]
        if len(self.field_names) == len(groups):
            for (a, v) in zip(self.field_names, groups):
                setattr(self, a, v)


class UnknownMatchLine(MatchLine):
    """
    A dummy class that represents a line that does not fit to any
    specified pattern

    """

    def __init__(self, line):
        self.line = line


class RawNote(MatchLine):
    """
    Class representing the played note part of a match line

    """
    field_names = ['Number', 'NoteName', 'Modifier', 'Octave',
                   'Onset', 'Offset', 'AdjOffset', 'Velocity']
    pattern = 'note\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)\)'
    re_obj = re.compile(pattern)

    def __init__(self, m):
        MatchLine.__init__(self, m)
        self.MidiPitch = pitch_name_2_midi_PC(
            self.Modifier, self.NoteName, self.Octave)


class RawNoteOld(MatchLine):
    """
    Class representing the played note part of a match line.
    Thi is for .matchfile version below 3.0.
    """
    field_names = ['Number', 'NoteName', 'Modifier', 'Octave',
                   'Onset', 'Offset', 'Velocity']
    pattern = 'note\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+),([^,]+),([^,]+)\)'
    re_obj = re.compile(pattern)

    def __init__(self, m):
        MatchLine.__init__(self, m)
        self.MidiPitch = pitch_name_2_midi_PC(
            self.Modifier, self.NoteName, self.Octave)


class Note(MatchLine):
    """
    Class representing the played note part of a match line

    """
    # field_names = ['Number', 'NoteName', 'Modifier', 'Octave',
    #                'Onset', 'Offset', 'AdjOffset', 'Velocity']
    # pattern = 'note\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)\)'
    # re_obj = re.compile(pattern)

    def __init__(self, m, version=LATEST_VERSION):

        self.version = version

        if self.version > 1:
            self.field_names = ['Number', 'NoteName', 'Modifier', 'Octave',
                                'Onset', 'Offset', 'AdjOffset', 'Velocity']
            self.pattern = 'note\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)\)'
        else:
            self.field_names = ['Number', 'NoteName', 'Modifier', 'Octave',
                                'Onset', 'Offset', 'Velocity']
            self.pattern = 'note\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+),([^,]+),([^,]+)\)'
        self.re_obj = re.compile(self.pattern)

        MatchLine.__init__(self, m)
        # super(Note, self).__init__(m)
        self.MidiPitch = pitch_name_2_midi_PC(
            self.Modifier, self.NoteName, self.Octave)


class TrailingNoteLine(MatchLine):
    """
    Class representing a Trailing Note line

    TODO
    ----
    * Check field names for old versions
    """
    field_names = ['Number', 'NoteName', 'Modifier', 'Octave',
                   'Onset', 'Offset', 'AdjOffset', 'Velocity']
    pattern = 'note\((.+),\[(.+),(.+)\],(.+),(.+),(.+),(.+),(.+)\)'
    re_obj = re.compile(pattern)

    def __init__(self, m, version=LATEST_VERSION):
        self.note = Note(m, version)


class Snote(MatchLine):
    """
    Class representing the score note part of a match line

    """
    field_names = ['Anchor', 'NoteName', 'Modifier', 'Octave',
                   'Bar', 'Beat', 'Offset', 'Duration',
                   'OnsetInBeats', 'OffsetInBeats', 'ScoreAttributesList']
    # pattern = 'snote\((.+),\[(.+),(.+)\],(.+),(.+):(.+),(.+),(.+),(.+),(.+),\[(.*)\]\)'
    pattern = 'snote\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+):([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),\[(.*)\]\)'
    re_obj = re.compile(pattern)

    def __init__(self, m=None):
        if m != None:
            MatchLine.__init__(self, m)
            self.DurationSymbolic = m.groups()[7]
            # import pdb
            # pdb.set_trace()
            try:
                self.ScoreAttributesList = self.ScoreAttributesList.split(',')
            except AttributeError:
                self.ScoreAttributesList = [self.ScoreAttributesList]

    @property
    def DurationInBeats(self):
        return self.OffsetInBeats - self.OnsetInBeats

    @property
    def MidiPitch(self):
        return pitch_name_2_midi_PC(self.Modifier, self.NoteName, self.Octave)


class InfoLine(MatchLine):
    """
    Class representing an Info line

    """
    field_names = ['Attribute', 'Value']
    pattern = 'info\(\s*([^,]+)\s*,\s*(.+)\s*\)\.'
    re_obj = re.compile(pattern)


class MetaLine(MatchLine):
    """
    Class representing a Meta line

    """
    field_names = ['Attribute', 'Value', 'Bar', 'TimeInBeats']
    pattern = 'meta\(\s*([^,]*)\s*,\s*([^,]*)\s*,\s*([^,]*)\s*,\s*([^,]*)\s*\)\.'
    re_obj = re.compile(pattern)


class SustainPedalLine(MatchLine):
    """
    Class representing a sustain pedal line

    """
    field_names = ['Time', 'Value']
    pattern = 'sustain\(\s*([^,]*)\s*,\s*([^,]*)\s*\)\.'
    re_obj = re.compile(pattern)


class SoftPedalLine(MatchLine):
    """
    Class representing a soft pedal line

    """
    field_names = ['Time', 'Value']
    pattern = 'soft\(\s*([^,]*)\s*,\s*([^,]*)\s*\)\.'
    re_obj = re.compile(pattern)


class SnoteNoteLine(MatchLine):
    """
    Class representing a "match" (containing snote and note)

    """
    # pattern = Snote.pattern+'-'+Note.pattern
    # re_obj = re.compile(pattern)

    def __init__(self, m1, m2, version=LATEST_VERSION):
        self.snote = Snote(m1)
        self.note = Note(m2, version)
        self.pattern = Snote.pattern + '-' + self.note.pattern
        self.re_obj = re.compile(self.pattern)


class SnoteDeletionLine(MatchLine):
    """
    Class representing the deletion of an snote

    """
    field_names = Snote.field_names
    pattern = Snote.pattern + '-deletion\.'  # unused for efficiency reasons
    re_obj = re.compile(pattern)

    def __init__(self, m1):
        self.snote = Snote(m1)


class InsertionNoteLine(MatchLine):
    # field_names = Note.field_names
    field_names = []
    # pattern = 'insertion-'+Note.pattern  # unused for efficiency reasons
    # re_obj = re.compile(pattern)

    def __init__(self, m2, version=LATEST_VERSION):
        self.note = Note(m2, version)
        self.pattern = 'insertion-' + self.note.pattern
        self.re_obj = re.compile(self.pattern)


class HammerBounceNoteLine(MatchLine):
    # field_names = Note.field_names
    # pattern = 'hammer_bounce-'+Note.pattern  # unused for efficiency reasons
    # re_obj = re.compile(pattern)

    def __init__(self, m2, version=LATEST_VERSION):
        self.note = Note(m2, version)
        self.field_names = self.note.field_names
        self.pattern = 'hammer_bounce-' + self.note.pattern
        self.re_obj = re.compile(self.pattern)


class OrnamentNoteLine(MatchLine):
    # field_names = Note.field_names
    # unused for efficiency reasons
    # pattern = 'ornament\([^\)]*\)-'+Note.pattern
    # re_obj = re.compile(pattern)

    def __init__(self, m2, version=LATEST_VERSION):
        self.note = Note(m2, version)
        self.field_names = self.note.field_names
        self.pattern = 'ornament\([^\)]*\)-' + self.note.pattern
        self.re_obj = re.compile(self.pattern)


class TrillNoteLine(MatchLine):
    # field_names = Note.field_names
    # pattern = 'trill\([^\)]*\)-'+Note.pattern  # unused for efficiency reasons
    # re_obj = re.compile(pattern)

    def __init__(self, m2, version=LATEST_VERSION):
        self.note = Note(m2, version)
        self.field_names = self.note.field_names
        self.pattern = 'trill\([^\)]*\)-' + self.note.pattern
        self.re_obj = re.compile(self.pattern)


class SnoteTrailingLine(MatchLine):
    field_names = ['Anchor', 'NoteName', 'Modifier', 'Octave',
                   'Bar', 'Beat', 'Offset', 'Duration',
                   'OnsetInBeats', 'OffsetInBeats', 'ScoreAttributesList']
    pattern = 'snote\((.+),\[(.+),(.+)\],(.+),(.+):(.+),(.+),(.+),(.+),(.+),\[(.*)\]\)'
    re_obj = re.compile(pattern)

    def __init__(self, m):
        self.snote = Snote(m)


class SnoteOnlyLine(MatchLine):
    field_names = ['Anchor', 'NoteName', 'Modifier', 'Octave',
                   'Bar', 'Beat', 'Offset', 'Duration',
                   'OnsetInBeats', 'OffsetInBeats', 'ScoreAttributesList']
    pattern = 'snote\((.+),\[(.+),(.+)\],(.+),(.+):(.+),(.+),(.+),(.+),(.+),\[(.*)\]\)'
    re_obj = re.compile(pattern)

    def __init__(self, m):
        self.snote = Snote(m)


class MatchFile(object):
    """
    Class for representing MatchFiles. It is instantiated by giving
    the filename of a Match file

    """

    def __init__(self, filename, version='auto'):
        """
        Read the contents of a Match file `filename`

        : param filename: filename of a match file

        """
        fileData = [l.decode('utf8').strip() for l in open(filename, 'rb')]

        self.name = filename

        self.voiceIdxFile = []

        self.version = LATEST_VERSION
        if version == 'auto':
            # print('Identifying Matchfile version')
            info_lines = [self.parse_matchline(l)
                          for l in fileData if l.startswith('info')]
            self.version = None
            for l in info_lines:
                if l.Attribute == 'matchFileVersion':
                    self.version = float(l.Value)
                    # print('Using version {0:.1f}'.format(self.version))

            if self.version is None:
                # print('File does include version. Using version {0:.1f}'.format(
                #     1.0))
                self.version = 1.0

        else:
            self.version = float(version)
            # print('Using version {0:.1f}'.format(self.version))
        # the lines of the file, represented as MatchLine objects
        self.lines = np.array([self.parse_matchline(l) for l in fileData])

    @property
    def _info(self):
        """
        Return all InfoLine objects

        """
        return [i for i in self.lines if isinstance(i, InfoLine)]

    def info(self, attribute=None):
        """
        Return the value of the InfoLine object corresponding to
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
    def sustain_lines(self):
        """
        Return all sustain pedal lines
        """

        return [i for i in self.lines if isinstance(i, SustainPedalLine)]

    @property
    def soft_lines(self):
        """
        Return all soft pedal lines
        """

        return [i for i in self.lines if isinstance(i, SoftPedalLine)]

    @property
    def note_pairs(self):
        """
        Return all(snote, note) tuples

        """
        return [(x.snote, x.note) for x in self.lines if isinstance(x, SnoteNoteLine)]

    def lines_at_score_times(self, times):
        """
        Return all lines with snotes that span any value t in the array `times`

        : param times: array of floats

        : returns: a list of MatchLine objects for each value t in `times`

        """
        snoteLines = [l for l in self.lines if hasattr(l, 'snote')]
        onoffsets = np.array([(l.snote.OnsetInBeats, l.snote.OffsetInBeats) for l in snoteLines],
                             dtype=np.dtype([('onset', np.float), ('offset', np.float)]))
        lidx = np.argsort(onoffsets, order=('onset', 'offset'))

        tidx = np.argsort(times)
        i = 0
        i_min = 0
        result = []
        for t in times[tidx]:
            r = []
            ii = []
            i = i_min
            while i < len(lidx) and not (onoffsets['onset'][lidx[i]] > t and onoffsets['offset'][lidx[i]] > t):
                if (onoffsets['onset'][lidx[i]] <= t and onoffsets['offset'][lidx[i]] > t):
                    r.append(lidx[i])
                    ii.append(i)
                i += 1
            if len(ii) > 0:
                i_min = ii[0]
            result.append(r)
        return [[snoteLines[x] for x in notes] for notes in result]

    @property
    def first_onset(self):
        """
        The earliest snote onset in the file

        """
        self.snote_idx()
        if len(self.snoteIdx) == 0:
            return None
        else:
            return self.lines[self.snoteIdx[0]].snote.OnsetInBeats

    @property
    def time_signatures(self):
        """
        A list of tuples(t, (a, b)), indicating a time signature of a over b, starting at t

        """
        tspat = re.compile('([0-9]+)/([0-9]*)')
        m = [(int(x[0]), int(x[1])) for x in
             tspat.findall(self.info('timeSignature'))]
        _timeSigs = []
        if len(m) > 0:
            _timeSigs.append((self.first_onset, m[0]))
        for l in self.time_sig_lines():
            _timeSigs.append((float(l.TimeInBeats), [
                            (int(x[0]), int(x[1])) for x in tspat.findall(l.Value)][0]))
        _timeSigs = list(set(_timeSigs))
        _timeSigs.sort(key=lambda x: x[0])

        # ensure that all consecutive time signatures are different
        timeSigs = [_timeSigs[0]]

        for ts in _timeSigs:
            ts_on, (ts_num, ts_den) = ts
            ts_on_prev, (ts_num_prev, ts_den_prev) = timeSigs[-1]
            if ts_num != ts_num_prev or ts_den != ts_den_prev:
                timeSigs.append(ts)

        return timeSigs

    def _time_sig_lines(self):
        return [i for i in self.lines if
                isinstance(i, MetaLine)
                and hasattr(i, 'Attribute')
                and i.Attribute == 'timeSignature']

    def time_sig_lines(self):
        ml = self._time_sig_lines()
        if len(ml) == 0:
            ts = self.info('timeSignature')
            ml = [self.parse_matchline(
                'meta(timeSignature,{0},1,{1}).'.format(ts, self.first_onset))]
        return ml

    def snote_idx(self):
        """
        Return the line numbers that have snotes

        """
        if hasattr(self, 'snotes'):
            return self.snoteIdx
        else:
            self.snoteIdx = [i for i, l in enumerate(self.lines)
                             if hasattr(l, 'snote')]
        return self.snoteIdx

    def soprano_voice(self, return_indices=False):
        """
        Return the snotes marked as soprano notes(excluding those
        marked as grace notes)

        : param return_indices: if True, return the line numbers of the
                               soprano notes, otherwise return the
                               corresponding MatchLines themselves

        : returns: a list of line numbers, or MatchLine objects

        """
        if return_indices:
            return [i for i, l in enumerate(self.lines)
                    if hasattr(l, 'snote')
                    and 's' in l.snote.ScoreAttributesList
                    and not 'grace' in l.snote.ScoreAttributesList
                    and l.snote.Duration > 0.0]
        else:
            return [l for l in self.lines
                    if hasattr(l, 'snote') and
                    's' in l.snote.ScoreAttributesList and
                    not 'grace' in l.snote.ScoreAttributesList and
                    l.snote.Duration > 0.0]

    def highest_voice_without_indexfile(self, exclude_grace=True, return_indices=False):
        """
        Return the highest snotes

        : param exclude_grace: if True, leave out any grace notes(default: True)
        : param return_indices: if True, return the line numbers of the soprano
                               notes, otherwise return the corresponding MatchLines
                               themselves(default: False)

        : returns: a list of line numbers, or MatchLine objects

        """
        sopr = self.soprano_voice(return_indices)

        if len(sopr) > 0:
            return(sopr)

        def is_grace(note):
            return 'grace' in note.ScoreAttributesList

        def in_lower_staff(note):
            return 'staff2' in note.ScoreAttributesList

        idx = self.snote_idx()

        features = []
        for i, idx in enumerate(self.snoteIdx):
            n = self.lines[idx].snote
            if not (in_lower_staff(n) or
                    (exclude_grace and is_grace(n)) or
                    n.Duration == 0.0):
                features.append(
                    (n.OnsetInBeats, n.OffsetInBeats, n.MidiPitch[0], i))

        features = np.array(features)
        # sort according to pitch (highest first)
        features = features[np.argsort(features[:, 2])[::-1]]

        # sort according to onset (smallest first)
        features = features[np.argsort(features[:, 0], kind='mergesort')]

        voice = [features[0, :]]
        for f in features:
            # if onset is later_eq than last voice offset, add next note
            if f[0] >= voice[-1][1]:
                voice.append(f)

        # indices into the list of snotes
        indices = np.array(np.array(voice)[:, 3], np.int)
        if return_indices:
            return np.array(self.snoteIdx)[indices]
        else:
            # return [m for i,m in enumerate(self.lines[self.snoteIdx]) if i in indices]
            return [l for l in self.lines[self.snoteIdx][indices]]

    def parse_matchline(self, l):
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

        or False if none can be matched

        """
        snoteMatch = Snote.match_pattern(l)

        if self.version > 1:
            noteMatch = RawNote.match_pattern(
                l, pos=snoteMatch.end() if snoteMatch else 0)
        else:
            noteMatch = RawNoteOld.match_pattern(
                l, pos=snoteMatch.end() if snoteMatch else 0)

        if snoteMatch:
            if noteMatch:
                return SnoteNoteLine(snoteMatch, noteMatch, version=self.version)
            else:
                if (re.compile('-deletion\.$').search(l, pos=snoteMatch.end())
                        or re.compile('-no_played_note\.$').search(l, pos=snoteMatch.end())):
                    return SnoteDeletionLine(snoteMatch)
                else:
                    if re.compile('-trailing_score_note\.$').search(l, pos=snoteMatch.end()):
                        return SnoteTrailingLine(snoteMatch)
                    else:
                        return SnoteOnlyLine(snoteMatch)
        else:  # no snoteMatch
            if noteMatch:
                if re.compile('^insertion-').search(l, endpos=noteMatch.start()):
                    return InsertionNoteLine(noteMatch, version=self.version)
                elif re.compile('^trill\([^\)]*\)-').search(l, endpos=noteMatch.start()):
                    return TrillNoteLine(noteMatch, version=self.version)
                elif re.compile('^ornament\([^\)]*\)-').search(l, endpos=noteMatch.start()):
                    return OrnamentNoteLine(noteMatch, version=self.version)
                elif re.compile('^trailing_played_note-').search(l, endpos=noteMatch.start()):
                    return TrailingNoteLine(noteMatch, version=self.version)
                elif re.compile('^hammer_bounce-').search(l, endpos=noteMatch.start()):
                    return HammerBounceNoteLine(noteMatch, version=self.version)
                else:
                    return False
            else:
                metaMatch = MetaLine.match_pattern(l)
                if metaMatch:
                    return MetaLine(metaMatch, lambda x: x)
                else:
                    infoMatch = InfoLine.match_pattern(l)
                    if infoMatch:
                        return InfoLine(infoMatch,
                                        field_interpreter=interpret_field)
                    else:
                        sustainMatch = SustainPedalLine.match_pattern(l)
                        if sustainMatch:
                            return SustainPedalLine(sustainMatch, field_interpreter=interpret_field)
                        else:
                            softMatch = SoftPedalLine.match_pattern(l)
                            if softMatch:
                                return SoftPedalLine(softMatch, field_interpreter=interpret_field)
                            else:
                                # return UnknownMatchLine(l)
                                return False


def match_to_notearray(fn_or_matchfile, sort_onsets=True,
                       expand_grace_notes=False,
                       score_attributes=[],
                       set_sustain=True,
                       pedal_threshold=63):
    """
    Extract the score and performance information from a MatchFile
    and makes a structured array

    Parameters
    ----------
    fn_or_matchfile : string or MatchFile
        Path to a .match file (string) or a Matchfile instance. If a path is
        given, the function reads the file and converts it into a Matchfile
        instance.
    sort_onsets : bool (optional)
        Sort array by onsets. This is the default option. Otherwise,
        the order of the notes will be the order in the matchfile
    expand_grace_notes: bool
        Expand the value of the grace notes.


    Returns
    -------
     score : structured array
        Structured array containing the score. The fields are
        'notes' (note name), 'octave', 'pitch', 'onset', 'duration',
        'velocity', 'p_onset', 'p_duration' plus any score attribute
        specified in `score_attributes`.
    """

    delete_grace_notes = False
    if isinstance(expand_grace_notes, str):

        if expand_grace_notes in ('omit', 'delete', 'd'):
            delete_grace_notes = True
            expand_grace_notes = False
        else:
            raise ValueError('`expand_grace_notes` must be a boolean or '
                             '"delete"')

    if isinstance(fn_or_matchfile, MatchFile):
        mf = fn_or_matchfile
    elif isinstance(fn_or_matchfile, str):
        # Parse matchfile
        mf = MatchFile(fn_or_matchfile)

    if set_sustain:
        adjust_offsets_w_sustain(mf, pedal_threshold)

    div = float(mf.info('midiClockUnits'))
    rate = float(mf.info('midiClockRate'))

    time_signatures = np.array([(ts[0], ts[1][0], ts[1][1])
                                for ts in mf.time_signatures])
    note_info = []
    c_ts = time_signatures[0, 1:]
    for snote, note in mf.note_pairs:
        ts_idx = np.where(time_signatures[:, 0] == snote.OnsetInBeats)[0]
        if len(ts_idx) > 0:
            c_ts = time_signatures[int(ts_idx), 1:]
        ni = get_note_info(snote=snote,
                           note=note, div=div,
                           rate=rate,
                           score_attributes=score_attributes,
                           use_adj_offset=set_sustain,
                           expand_grace_notes=expand_grace_notes,
                           ts=c_ts)
        note_info.append(ni)

    # output dtypes
    dtypes = [('notes', 'S8'),
              ('octave', 'i4'),
              ('pitch', 'i4'),
              ('onset', 'f4'),
              ('duration', 'f4'),
              ('velocity', 'f4'),
              ('p_onset', 'f4'),
              ('p_duration', 'f4'),
              ('bar', 'i4'),
              ('onset_in_bar', 'f4'),
              ('beat_phase', 'f4')]

    for attr in score_attributes:
        dtypes.append((attr, 'i4'))
    note_info = np.array(note_info, dtype=dtypes)

    if delete_grace_notes:
        # Remove notes with duration equal to 0
        note_info = note_info[note_info['duration'] != 0]
    if sort_onsets:
        sort_idx_pitch = np.argsort(note_info['pitch'])
        note_info = note_info[sort_idx_pitch]
        # use a stable sorting to sort by onset and then by pitch
        sort_idx_onset = np.argsort(note_info['onset'], kind='mergesort')
        note_info = note_info[sort_idx_onset]

    return note_info


def get_note_info(snote, note, div=480.0, rate=50000.0, score_attributes=[],
                  use_adj_offset=True, expand_grace_notes=True,
                  ts=(4, 4)):
    """
    Get note information from a `Snote` instance

    Parameters
    ----------
    snote : `matchfile.Snote`
        Instance of `Snote` from the MatchFile` containing
        the score note information
    note : `matchfile.Note`
        Performed note information
    div = float
        Time division
    score_attributes : list
        A list of strings with the name of score attributes

    Returns
    -------
    note_name : str
        Name of the note
    octave : int
        Octave of the note
    pitch : int
        MIDI pitch
    onset_b : float
        Onset in beats
    duration_b : float
        Duration in beats
    velocity : int
        Performed MIDI velocity
    onset_s : float
        Performed onset in seconds
    duration_s : float
        Performed duration in seconds
    score_attributes : bool
        If `score_attributes` is not an empty list, it appends as
        many booleans indicating whether the note has a score attribute.
    """
    # Name of the note (wo accidental)
    step = str(snote.NoteName).upper()
    # accidental
    modifier = str(snote.Modifier)
    onset_b = snote.OnsetInBeats
    offset_b = snote.OffsetInBeats
    octave = snote.Octave
    pitch = snote.MidiPitch[0]
    duration_b = offset_b - onset_b
    ts_num, ts_den = ts
    bar = snote.Bar
    print(snote.Beat ,  snote.Offset,  ts_den)
    onset_in_bar = float(snote.Beat + snote.Offset * ts_den)
    beat_phase = np.mod(onset_in_bar, ts_num) / float(ts_num)

    # Check for score attributes
    out_attributes = []

    for attribute in score_attributes:
        att = 0
        if attribute in snote.ScoreAttributesList:
            att = 1

        out_attributes.append(att)

    is_grace = duration_b == 0 or 'grace' in snote.ScoreAttributesList

    if is_grace and expand_grace_notes:
        if duration_b == 0:
            # This is a hack. This just adds a small value to the
            # duration of the note and shifts the onset an eight
            # of a beat.
            onset_b -= 0.125
            duration_b = 0.125

    # TODO:
    # * Check if this case is correct for bb and x
    if modifier != 'n':
        step += modifier

    velocity = note.Velocity
    onset_s = float(note.Onset) * rate / (div * 1e6)

    if use_adj_offset:
        # Get adjusted offset if available, else use use the unadjusted
        offset_in_ticks = max(float(getattr(note, 'AdjOffset', note.Offset)),
                              note.Offset)

        if offset_in_ticks < note.Onset:
            print('Negative duration! The duration of the note is set to 1')
            offset_in_ticks = note.Onset + 1

    else:
        offset_in_ticks = float(note.Offset)
    offset_s = offset_in_ticks * rate / (div * 1e6)

    duration_s = offset_s - onset_s

    return (step, octave, pitch, onset_b, duration_b, velocity,
            onset_s, duration_s, bar, onset_in_bar, beat_phase) + tuple(out_attributes)


def adjust_offsets_w_sustain(mf, threshold=63):
    """
    Add sustain pedal information if available to the adjusted offset
    """
    # get all performed notes
    notes = [l.note for l in mf.lines if hasattr(l, 'note')]

    # get all note offsets
    offs = np.array([n.Offset for n in notes])
    first_off = np.min(offs)
    last_off = np.max(offs)

    sustain_lines = mf.sustain_lines

    if len(sustain_lines) > 0:
        # Get pedal times
        pedal = np.array([(x.Time, x.Value > threshold) for x in sustain_lines])
        # sort, just in case
        pedal = pedal[np.argsort(pedal[:, 0]), :]

        # reduce the pedal info to just the times where there is a change in pedal state
        pedal = np.vstack(((min(pedal[0, 0] - 1, first_off - 1), 0),  # if there is an onset before the first pedal info, assume pedal is off
                           pedal[0, :],
                           pedal[np.where(np.diff(pedal[:, 1]) != 0)[0] + 1, :],
                           (max(pedal[-1, 0] + 1, last_off + 1), 0)  # if there is an offset after the last pedal info, assume pedal is off
                           ))
        last_pedal_change_before_off = np.searchsorted(pedal[:, 0], offs) - 1

        pedal_state_at_off = pedal[last_pedal_change_before_off, 1]
        pedal_down_at_off = pedal_state_at_off == 1
        next_pedal_time = pedal[last_pedal_change_before_off + 1, 0]

        offs[pedal_down_at_off] = next_pedal_time[pedal_down_at_off]

        for offset, note in zip(offs, notes):
            note.AdjOffset = offset


if __name__ == '__main__':
    pass
