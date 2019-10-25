#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d

from . import find_nearest, search, iter_current_next

MIDI_CONTROL_TYPES = {
    64: 'sustain_pedal',
    67: 'soft_pedal'
    }
MIDI_BASE_CLASS = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}
# _MORPHETIC_BASE_CLASS = {'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6}
# _MORPHETIC_OCTAVE = {0: 32, 1: 39, 2: 46, 3: 53, 4: 60, 5: 67, 6: 74, 7: 81, 8: 89}
ALTER_SIGNS = {None: '', 0: '', 1: '#', 2: 'x', -1: 'b', -2: 'bb'}

LABEL_DURS = {
    'long': 16,
    'breve': 8,
    'whole': 4,
    'half': 2,
    'h': 2,
    'quarter': 1,
    'q': 1,
    'eighth': 1 / 2,
    'e': 1 / 2,
    '16th': 1 / 4,
    '32nd': 1 / 8.,
    '64th': 1 / 16,
    '128th': 1 / 32,
    '256th': 1 / 64
}
DOT_MULTIPLIERS = (1, 1 + 1 / 2, 1 + 3 / 4, 1 + 7 / 8)
# DURS and SYM_DURS encode the same information as _LABEL_DURS and
# _DOT_MULTIPLIERS, but they allow for faster estimation of symbolic duration
# (estimate_symbolic duration). At some point we will probably do away with
# _LABEL_DURS and _DOT_MULTIPLIERS.
DURS = np.array([
    1.5625000e-02, 2.3437500e-02, 2.7343750e-02, 2.9296875e-02, 3.1250000e-02,
    4.6875000e-02, 5.4687500e-02, 5.8593750e-02, 6.2500000e-02, 9.3750000e-02,
    1.0937500e-01, 1.1718750e-01, 1.2500000e-01, 1.8750000e-01, 2.1875000e-01,
    2.3437500e-01, 2.5000000e-01, 3.7500000e-01, 4.3750000e-01, 4.6875000e-01,
    5.0000000e-01, 5.0000000e-01, 7.5000000e-01, 7.5000000e-01, 8.7500000e-01,
    8.7500000e-01, 9.3750000e-01, 9.3750000e-01, 1.0000000e+00, 1.0000000e+00,
    1.5000000e+00, 1.5000000e+00, 1.7500000e+00, 1.7500000e+00, 1.8750000e+00,
    1.8750000e+00, 2.0000000e+00, 2.0000000e+00, 3.0000000e+00, 3.0000000e+00,
    3.5000000e+00, 3.5000000e+00, 3.7500000e+00, 3.7500000e+00, 4.0000000e+00,
    6.0000000e+00, 7.0000000e+00, 7.5000000e+00, 8.0000000e+00, 1.2000000e+01,
    1.4000000e+01, 1.5000000e+01, 1.6000000e+01, 2.4000000e+01, 2.8000000e+01,
    3.0000000e+01])

SYM_DURS = [
    {"type": "256th", "dots": 0},
    {"type": "256th", "dots": 1},
    {"type": "256th", "dots": 2},
    {"type": "256th", "dots": 3},
    {"type": "128th", "dots": 0},
    {"type": "128th", "dots": 1},
    {"type": "128th", "dots": 2},
    {"type": "128th", "dots": 3},
    {"type": "64th", "dots": 0},
    {"type": "64th", "dots": 1},
    {"type": "64th", "dots": 2},
    {"type": "64th", "dots": 3},
    {"type": "32nd", "dots": 0},
    {"type": "32nd", "dots": 1},
    {"type": "32nd", "dots": 2},
    {"type": "32nd", "dots": 3},
    {"type": "16th", "dots": 0},
    {"type": "16th", "dots": 1},
    {"type": "16th", "dots": 2},
    {"type": "16th", "dots": 3},
    {"type": "eighth", "dots": 0},
    {"type": "e", "dots": 0},
    {"type": "eighth", "dots": 1},
    {"type": "e", "dots": 1},
    {"type": "eighth", "dots": 2},
    {"type": "e", "dots": 2},
    {"type": "eighth", "dots": 3},
    {"type": "e", "dots": 3},
    {"type": "quarter", "dots": 0},
    {"type": "q", "dots": 0},
    {"type": "quarter", "dots": 1},
    {"type": "q", "dots": 1},
    {"type": "quarter", "dots": 2},
    {"type": "q", "dots": 2},
    {"type": "quarter", "dots": 3},
    {"type": "q", "dots": 3},
    {"type": "half", "dots": 0},
    {"type": "h", "dots": 0},
    {"type": "half", "dots": 1},
    {"type": "h", "dots": 1},
    {"type": "half", "dots": 2},
    {"type": "h", "dots": 2},
    {"type": "half", "dots": 3},
    {"type": "h", "dots": 3},
    {"type": "whole", "dots": 0},
    {"type": "whole", "dots": 1},
    {"type": "whole", "dots": 2},
    {"type": "whole", "dots": 3},
    {"type": "breve", "dots": 0},
    {"type": "breve", "dots": 1},
    {"type": "breve", "dots": 2},
    {"type": "breve", "dots": 3},
    {"type": "long", "dots": 0},
    {"type": "long", "dots": 1},
    {"type": "long", "dots": 2},
    {"type": "long", "dots": 3}
]

MAJOR_KEYS = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#']
MINOR_KEYS = ['Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#']


def pitch_spelling_to_midi_pitch(step, alter, octave):
    midi_pitch = ((octave + 1) * 12 +
                  MIDI_BASE_CLASS[step.lower()] +
                  (alter or 0))
    return midi_pitch


SIGN_TO_ALTER = {'n': 0, '#': 1, 'x': 2, '##': 2, '###': 3,
                 'b': -1, 'bb': -2, 'bbb': -3, '-': None}


def ensure_pitch_spelling_format(step, alter, octave):

    if step.lower() not in MIDI_BASE_CLASS:
        if step.lower() != 'r':
            raise ValueError('Invalid `step`')

    if isinstance(alter, str):
        try:
            alter = SIGN_TO_ALTER[alter]
        except KeyError:
            raise ValueError('Invalid `alter`, must be ("n", "#", "x", "b" or "bb"), but given {0}'.format(alter))

    if not isinstance(alter, int):
        try:
            alter = int(alter)
        except TypeError or ValueError:
            if alter is not None:
                raise ValueError('`alter` must be an integer or None')

    if octave == '-':
        # check octave for weird rests in Batik match files
        octave = None
    else:
        if not isinstance(octave, int):
            try:
                octave = int(octave)
            except TypeError or ValueError:
                if octave is not None:
                    raise ValueError('`octave` must be an integer or None')

    return step.upper(), alter, octave


def fifths_mode_to_key_name(fifths, mode=None):
    """Return the key signature name corresponding to a number of sharps
    or flats and a mode. A negative value for `fifths` denotes the
    number of flats (i.e. -3 means three flats), and a positive
    number the number of sharps. The mode is specified as 'major'
    or 'minor'. If `mode` is None, the key is assumed to be major.

    Parameters
    ----------
    fifths : int
        Number of fifths
    mode : {'major', 'minor', None}
        Mode of the key signature

    Returns
    -------
    str
        The name of the key signature, e.g. 'Am'

    Examples
    --------
    >>> fifths_mode_to_key_name(0, 'minor')
    'Am'
    >>> fifths_mode_to_key_name(0, 'major')
    'C'
    >>> fifths_mode_to_key_name(3, 'major')
    'A'

    """
    global MAJOR_KEYS, MINOR_KEYS

    if mode == 'minor':
        keylist = MINOR_KEYS
        suffix = 'm'
    elif mode in ('major', None):
        keylist = MAJOR_KEYS
        suffix = ''
    else:
        raise Exception('Unknown mode {}'.format(mode))

    try:
        name = keylist[fifths + 7]
    except IndexError:
        raise Exception('Unknown number of fifths {}'.format(fifths))

    return name + suffix


def key_name_to_fifths_mode(name):
    """Return the number of sharps or flats and the mode of a key
    signature name. A negative number denotes the number of flats
    (i.e. -3 means three flats), and a positive number the number of
    sharps. The mode is specified as 'major' or 'minor'.

    Parameters
    ----------
    name : {"A", "A#m", "Ab", "Abm", "Am", "B", "Bb", "Bbm", "Bm", "C", "C#",
        "C#m", "Cb", "Cm", "D", "D#m", "Db", "Dm", "E", "Eb", "Ebm",
        "Em", "F", "F#", "F#m", "Fm", "G", "G#m", "Gb", "Gm"} Name of
        the key signature

    Returns
    -------


    Examples
    --------
    >>> key_name_to_fifths_mode('Am')
    (0, 'minor')
    >>> key_name_to_fifths_mode('C')
    (0, 'major')
    >>> key_name_to_fifths_mode('A')
    (3, 'major')

    """
    global MAJOR_KEYS, MINOR_KEYS

    if name.endswith('m'):
        mode = 'minor'
        keylist = MINOR_KEYS
    else:
        mode = 'major'
        keylist = MAJOR_KEYS

    try:
        fifths = keylist.index(name.strip('m')) - 7
    except ValueError:
        raise Exception('Unknown key signature {}'.format(name))

    return fifths, mode


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
    >>> estimate_symbolic_duration(15, 10)
    {'type': 'quarter', 'dots': 1}

    The following example returns None:

    >>> estimate_symbolic_duration(23, 16)


    Returns
    -------
    dict or None
        A dictionary containing keys 'type' and 'dots' expressing the estimated
        symbolic duration or None
    """
    global DURS, SYM_DURS
    qdur = dur / div
    i = find_nearest(DURS, qdur)
    if np.abs(qdur - DURS[i]) < eps:
        return SYM_DURS[i].copy()
    else:
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
    return float(tempo * DOT_MULTIPLIERS[dots] * LABEL_DURS[unit])


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
        result = (symbolic_dur.get('type') or '') + '.' * symbolic_dur.get('dots', 0)

        if 'actual_notes' in symbolic_dur and 'normal_notes' in symbolic_dur:

            result += '_{}/{}'.format(symbolic_dur['actual_notes'],
                                      symbolic_dur['normal_notes'])

        return result


def symbolic_to_numeric_duration(symbolic_dur, divs):
    numdur = divs * LABEL_DURS[symbolic_dur.get('type', None)]
    numdur *= DOT_MULTIPLIERS[symbolic_dur.get('dots', 0)]
    numdur *= ((symbolic_dur.get('normal_notes') or 1) /
               (symbolic_dur.get('actual_notes') or 1))
    return numdur


def order_splits(start, end, smallest_unit):
    """Description

    Parameters
    ----------
    start : int
        Description of `start`
    end : int
        Description of `end`
    smallest_divs : int
        Description of `smallest_divs`

    Returns
    -------
    ndarray
        Description of return value

    Examples
    --------
    >>> order_splits(1, 8, 1)
    array([4, 2, 6, 3, 5, 7])
    >>> order_splits(11, 17, 3)
    array([12, 15])
    >>> order_splits(11, 17, 1)
    array([16, 12, 14, 13, 15])
    >>> order_splits(11, 17, 4)
    array([16, 12])

    """

    # gegeven b, kies alle veelvouden van 2*b, verschoven om b, die tussen start en end liggen
    # gegeven b, kies alle veelvouden van 2*b die tussen start-b en end-b liggen en tel er b bij op

    b = smallest_unit
    result = []

    splits = np.arange((b * 2) * (1 + (start + b) // (b * 2)), end + b, b * 2) - b

    while b * (1 + start // b) < end and b * (end // b) > start:
        result.insert(0, splits)
        b = b * 2
        splits = np.arange((b * 2) * (1 + (start + b) // (b * 2)), end + b, b * 2) - b

    if result:
        return np.concatenate(result)
    else:
        return np.array([])


def find_smallest_unit(divs):
    unit = divs
    while unit % 2 == 0:
        unit = unit // 2
    return unit


def find_tie_split(start, end, divs, max_splits=3):
    """
    Examples
    --------

    >>> find_tie_split_search(1, 8, 2)
    [(1, 8, {'type': 'half', 'dots': 2})]

    >>> find_tie_split_search(0, 3615, 480)
    [(0, 3600, {'type': 'whole', 'dots': 3}), (3600, 3615, {'type': '128th', 'dots': 0})]

    """

    smallest_unit = find_smallest_unit(divs)

    def success(state):
        return all(estimate_symbolic_duration(right - left, divs)
                   for left, right in iter_current_next([start] + state + [end]))

    def expand(state):
        if len(state) >= max_splits:
            return []
        else:
            split_start = ([start] + state)[-1]
            ordered_splits = order_splits(split_start, end, smallest_unit)
            new_states = [state + [s.item()] for s in ordered_splits]
            # start and end must be "in sync" with splits for states to succeed
            new_states = [s for s in new_states if
                          (s[0] - start) % smallest_unit == 0
                          and (end - s[-1]) % smallest_unit == 0]
            return new_states

    def combine(new_states, old_states):
        return old_states + new_states

    states = [[]]

    # splits = search_recursive(states, success, expand, combine)
    splits = search(states, success, expand, combine)

    if splits is not None:
        solution = [(left, right, estimate_symbolic_duration(right - left, divs))
                    for left, right in iter_current_next([start] + splits + [end])]
        # print(solution)
        return solution
    else:
        pass  # print('no solution for ', start, end, divs)


def estimate_clef_properties(pitches):
    # estimate the optimal clef for the given pitches. This returns a dictionary
    # with the sign, line, and octave_change attributes of the clef
    # (cf. MusicXML clef description). Currently only G and F clefs without
    # octave changes are supported.

    center = np.median(pitches)
    # number, sign, line, octave_change):
    # clefs = [score.Clef(1, 'F', 4, 0), score.Clef(1, 'G', 2, 0)]
    clefs = [dict(sign='F', line=4, octave_change=0),
             dict(sign='G', line=2, octave_change=0)]
    f = interp1d([0, 49, 70, 127], [0, 0, 1, 1], kind='nearest')
    return clefs[int(f(center))]


def notes_to_notearray(notes):
    # create a structured array with fields pitch, onset and duration for a
    # given list of notes
    return np.array([(n.midi_pitch, n.start.t, n.duration_tied) for n in notes],
                    dtype=[('pitch', 'i4'), ('onset', 'i4'), ('duration', 'i4')])


if __name__ == '__main__':
    import doctest
    doctest.testmod()
