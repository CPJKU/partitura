#!/usr/bin/env python
from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix

from partitura.utils.generic import find_nearest, search, iter_current_next

MIDI_BASE_CLASS = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}
# _MORPHETIC_BASE_CLASS = {'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6}
# _MORPHETIC_OCTAVE = {0: 32, 1: 39, 2: 46, 3: 53, 4: 60, 5: 67, 6: 74, 7: 81, 8: 89}
ALTER_SIGNS = {None: '', 0: '', 1: '#', 2: 'x', -1: 'b', -2: 'bb'}

DUMMY_PS_BASE_CLASS = {0: ('c', 0),
                       1: ('c', 1),
                       2: ('d', 0),
                       3: ('d', 1),
                       4: ('e', 0),
                       5: ('f', 0),
                       6: ('f', 1),
                       7: ('g', 0),
                       8: ('g', 1),
                       9: ('a', 0),
                       10: ('a', 1),
                       11: ('b', 0)}

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


def ensure_notearray(notearray_or_part):
    """
    Ensures to get a structured note array from the input.

    Parameters
    ----------
    notearray_or_part : structured ndarray, `Part` or `PerformedPart`
        Input score information

    Returns
    -------
    structured ndarray
        Structured array containing score information.
    """
    from partitura.score import Part, PartGroup
    from partitura.performance import PerformedPart

    if isinstance(notearray_or_part, np.ndarray):
        if notearray_or_part.dtype.fields is not None:
            return notearray_or_part
        else:
            raise ValueError('Input array is not a structured array!')
    elif isinstance(notearray_or_part, (Part, PartGroup, PerformedPart)):
        return notearray_or_part.note_array
    else:
        raise ValueError('`notearray_or_part` should be a structured '
                         'numpy array, a `Part`, `PartGroup` or a '
                         '`PerformedPart`, but '
                         'is {0}'.format(type(notearray_or_part)))


def pitch_spelling_to_midi_pitch(step, alter, octave):
    midi_pitch = ((octave + 1) * 12 +
                  MIDI_BASE_CLASS[step.lower()] +
                  (alter or 0))
    return midi_pitch


def midi_pitch_to_pitch_spelling(midi_pitch):
    octave = midi_pitch // 12 - 1
    step, alter = DUMMY_PS_BASE_CLASS[np.mod(midi_pitch, 12)]
    return ensure_pitch_spelling_format(step, alter, octave)


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
    mode : {'major', 'minor', None, -1, 1}
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
    >>> fifths_mode_to_key_name(-1, 1)
    'F'

    """
    global MAJOR_KEYS, MINOR_KEYS

    if mode in ('minor', -1):
        keylist = MINOR_KEYS
        suffix = 'm'
    elif mode in ('major', None, 1):
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
    name : {"A", "A#m", "Ab", "Abm", "Am", "B", "Bb", "Bbm", "Bm", "C", "C#", "C#m", "Cb", "Cm", "D", "D#m", "Db", "Dm", "E", "Eb", "Ebm", "Em", "F", "F#", "F#m", "Fm", "G", "G#m", "Gb", "Gm"}
        Name of the key signature

    Returns
    -------
    (int, str)
        Tuple containing the number of fifths and the mode 


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

def key_mode_to_int(mode):
    """Return the mode of a key as an integer (1 for major and -1 for
    minor).

    Parameters
    ----------
    mode : {'major', 'minor', None, 1, -1}
        Mode of the key

    Returns
    -------
    int
        Integer representation of the mode.

    """
    if mode in ('minor', -1):
        return -1
    elif mode in ('major', None, 1):
        return 1
    else:
        raise ValueError('Unknown mode {}'.format(mode))

def key_int_to_mode(mode):
    """Return the mode of a key as a string ('major' or 'minor')

    Parameters
    ----------
    mode : {'major', 'minor', None, 1, -1}
        Mode of the key

    Returns
    -------
    int
        Integer representation of the mode.

    """
    if mode in ('minor', -1):
        return 'minor'
    elif mode in ('major', None, 1):
        return 'major'
    else:
        raise ValueError('Unknown mode {}'.format(mode))


def estimate_symbolic_duration(dur, div, eps=10**-3):
    """Given a numeric duration, a divisions value (specifiying the
    number of units per quarter note) and optionally a tolerance `eps`
    for numerical imprecisions, estimate corresponding the symbolic
    duration. If a matching symbolic duration is found, it is returned
    as a tuple (type, dots), where type is a string such as 'quarter',
    or '16th', and dots is an integer specifying the number of dots.
    If no matching symbolic duration is found the function returns
    None.

    NOTE : this function does not estimate composite durations, nor
    time-modifications such as triplets.

    Parameters
    ----------
    dur : float or int
        Numeric duration value
    div : int
        Number of units per quarter note
    eps : float, optional (default: 10**-3)
        Tolerance in case of imprecise matches

    Returns
    -------


    Examples
    --------
    >>> estimate_symbolic_duration(24, 16)
    {'type': 'quarter', 'dots': 1}

    >>> estimate_symbolic_duration(15, 10)
    {'type': 'quarter', 'dots': 1}

    The following example returns None:

    >>> estimate_symbolic_duration(23, 16)

    """
    global DURS, SYM_DURS
    qdur = dur / div
    i = find_nearest(DURS, qdur)
    if np.abs(qdur - DURS[i]) < eps:
        return SYM_DURS[i].copy()
    else:
        return None


def to_quarter_tempo(unit, tempo):
    """Given a string `unit` (e.g. 'q', 'q.' or 'h') and a number
    `tempo`, return the corresponding tempo in quarter notes. This is
    useful to convert textual tempo directions like h=100.

    Parameters
    ----------
    unit : str
        Tempo unit
    tempo : number
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
    """Create a string representation of the symbolic duration encoded
    in the dictionary `symbolic_dur`.

    Parameters
    ----------
    symbolic_dur : dict
        Dictionary with keys 'type' and 'dots'

    Returns
    -------
    str
        A string representation of the specified symbolic duration

    Examples
    --------
    >>> format_symbolic_duration({'type': 'q', 'dots': 2})
    'q..'

    >>> format_symbolic_duration({'type': '16th'})
    '16th'

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


def notearray_to_pianoroll(note_array, time_div=8, onset_only=False,
                           note_separation=False,
                           pitch_margin=-1, time_margin=0,
                           return_idxs=False,
                           is_performance=False,
                           piano_range=False):
    """Computes a piano roll from a structured note array (as
    generated by the `note_array` methods in `partitura.score.Part`
    and `partitura.performance.PerformedPart` instances).

    Parameters
    ----------
    notes : structured array
        Structured array with pitch, onset and duration information.
        If the array represents a score is expected to have field
        names "pitch", "onset" and "duration" (as would be generated
        by the property `note_array`from a `partitura.score.Part`
        instance). If the array represents a performance, it is
        expected to have fields "pitch", "p_onset", "p_duration" and
        "velocity" (as would be generated by property `note_array`
        from  an instance of `partitura.performance.PerformedPart`).
    time_div : int, optional
        How many sub-divisions for each time unit (beats for a score
        or seconds for a performance. See `is_performance` below).
    onset_only : bool, optional
        If True, code only the onsets of the notes, otherwise code
        onset and duration.
    pitch_margin : int, optional
        If `pitch_margin` > -1, the resulting array will have
        `pitch_margin` empty rows above and below the highest and
        lowest pitches, respectively; if `pitch_margin` == -1, the
        resulting pianoroll will have span the fixed pitch range
        between (and including) 1 and 127.
    time_margin : int, optional
        The resulting array will have `time_margin` * `time_div` empty
        columns before and after the piano roll
    return_idxs : bool, optional
        If True, return the indices (i.e., the coordinates) of each
        note in the piano roll.
    is_performance : bool, optional
        If True, assumes that the note array is not a score (i.e., a
        performance), and the time unit is in seconds insted of beats.
    piano_range : bool, optional
        If True, the pitch axis of the piano roll is in piano keys
        instead of MIDI note numbers (and there are only 88 pitches).
        This is equivalent as slicing `piano_range_pianoroll =
        pianoroll[21:109, :]`.

    Returns
    -------
    pianoroll : scipy.sparse.csr_matrix
        A sparse int matrix of size representing the pianoroll; The
        first dimension is pitch, the second is time; The sizes of the
        dimensions vary with the parameters `pitch_margin`,
        `time_margin`, and `time_div`
    pr_idx : ndarray
        Indices of the onsets and offsets of the notes in the piano
        roll (in the same order as the input note_array). This is only
        returned if `return_idxs` is `True`.

    Examples
    --------

    >>> import numpy as np
    >>> note_array = np.array([(60, 0, 1)],
                          dtype=[('pitch', 'i4'),
                                 ('onset', 'f4'),
                                 ('duration', 'f4')])

    >>> pr = notearray_to_pianoroll(note_array, pitch_margin=2, time_div=2)
    >>> pr.toarray()
    array([[0, 0],
          [0, 0],
          [1, 1],
          [0, 0],
          [0, 0]])

    Notes
    -----
    The default values in this function assume that the input
    `note_array` represents a score.

    """
    # Fields for performance
    p_fields = ['pitch', 'p_onset', 'p_duration', 'velocity']
    # Fields for score
    s_fields = ['pitch', 'onset', 'duration']

    fields = p_fields if is_performance else s_fields

    for field in fields:
        if field not in note_array.dtype.names:
            raise ValueError(
                '`note_array` does not have field `{0}`'.format(field))

    if time_div < 1:
        raise ValueError('`time_div` should be larger than 1.')

    note_info = np.column_stack([note_array[field].astype(np.float)
                                 for field in fields])
    return _notearray_to_pianoroll(note_info=note_info,
                                   time_div=time_div,
                                   onset_only=onset_only,
                                   note_separation=note_separation,
                                   pitch_margin=pitch_margin,
                                   time_margin=time_margin,
                                   return_idxs=return_idxs,
                                   piano_range=piano_range)


def _notearray_to_pianoroll(note_info, onset_only=False,
                            pitch_margin=-1, time_margin=0,
                            time_div=8, note_separation=True,
                            return_idxs=False,
                            piano_range=False):
    # non-public
    """Computes a piano roll from a numpy array with MIDI pitch,
    onset, duration and (optionally) MIDI velocity information. See
    `notearray_to_pianoroll` for a complete description of the
    arguments of this function.

    """

    # Get pitch, onset, offset from the note_info array
    pr_pitch = note_info[:, 0]
    onset = note_info[:, 1]
    offset = note_info[:, 1] + note_info[:, 2]

    # Get velocity if given
    if note_info.shape[1] < 4:
        pr_velocity = np.ones(len(note_info))
    else:
        pr_velocity = note_info[:, 3]

    # Adjust pitch margin
    if pitch_margin > -1:
        highest_pitch = np.max(pr_pitch)
        lowest_pitch = np.min(pr_pitch)
    else:
        lowest_pitch = 0
        highest_pitch = 127

    pitch_span = highest_pitch - lowest_pitch + 1

    # sorted idx
    idx = np.argsort(onset)
    # sort notes
    pr_pitch = pr_pitch[idx]
    onset = onset[idx]
    offset = offset[idx]
    min_time = onset[0]
    max_time = np.max(offset)

    onset -= min_time - time_margin
    offset -= min_time - time_margin

    if pitch_margin > -1:
        pr_pitch -= lowest_pitch
        pr_pitch += pitch_margin

    # Size of the output piano roll
    # Pitch dimension
    if pitch_margin > -1:
        M = int(pitch_span + 2 * pitch_margin)
    else:
        M = int(pitch_span)

    # Time dimension
    N = int(np.ceil(time_div * (2 * time_margin + max_time - min_time)))

    # Onset and offset times of the notes in the piano roll
    pr_onset = np.round(time_div * onset).astype(np.int)
    pr_offset = np.round(time_div * offset).astype(np.int)

    # Determine the non-zero indices of the piano roll
    if onset_only:
        _idx_fill = np.column_stack([pr_pitch, pr_onset, pr_velocity])
    else:
        pr_offset = np.maximum(pr_onset + 1,
                               pr_offset - (1 if note_separation else 0))
        _idx_fill = np.vstack([np.column_stack((np.zeros(off - on) + pitch,
                                               np.arange(on, off),
                                               np.zeros(off - on) + vel))
                              for on, off, pitch, vel in zip(pr_onset,
                                                             pr_offset,
                                                             pr_pitch,
                                                             pr_velocity)
                              if off <= N])

    # Fix multiple notes with the same pitch and onset
    fill_dict = defaultdict(list)
    for row, col, vel in _idx_fill:
        key = (int(row), int(col))
        fill_dict[key].append(vel)

    idx_fill = np.zeros((len(fill_dict), 3))
    for i, ((row, column), vel) in enumerate(fill_dict.items()):
        idx_fill[i] = np.array([row, column, max(vel)])
    
    # Fill piano roll
    pianoroll = csc_matrix((idx_fill[:, 2],
                            (idx_fill[:, 0], idx_fill[:, 1])),
                           shape=(M, N), dtype=np.int)

    pr_idx_pitch_start = 0
    if piano_range:
        pianoroll = pianoroll[21:109, :]
        pr_idx_pitch_start = 21

    if return_idxs:
        # indices of each note in the piano roll
        pr_idx = np.column_stack([pr_pitch - pr_idx_pitch_start,
                                  pr_onset, pr_offset]).astype(np.int)
        return pianoroll, pr_idx[idx.argsort()]
    else:
        return pianoroll


def pianoroll_to_notearray(pianoroll, time_div=8):
    """Extract a structured note array from a piano roll.

    For now, the structured note array is considered a
    "performance".

    Parameters
    ----------
    pianoroll : array-like
        2D array containing a piano roll. The first dimension is
        pitch, and the second is time. The value of each "pixel" in
        the piano roll is considered to be the MIDI velocity, and it
        is supposed to be between 0 and 127.
    time_div : int
        How many sub-divisions for each time unit (see
        `notearray_to_pianoroll`).

    Returns
    -------
    np.ndarray :
        Structured array with pitch, onset, duration and velocity
        fields.

    Notes
    -----
    Please note that all non-zero pixels will contribute to a note.
    For the case of piano rolls with continuous values between 0 and 1
    (as might be the case of those piano rolls produced using
    probabilistic/generative models), we recomend to either 1) hard-
    threshold the piano roll to have only 0s (note-off) or 1s (note-
    on) or, 2) soft-threshold the notes (values below a certain
    threshold are considered as not active and scale the active notes
    to lie between 1 and 127).

    """
    # Indices of the non-zero elements of the piano roll
    pitch_idx, active_idx = pianoroll.nonzero()

    # Sort indices according to time and pitch
    time_sort_idx = np.argsort(active_idx)
    pitch_sort_idx = pitch_idx[time_sort_idx].argsort(kind='mergesort')

    pitch_idx = pitch_idx[time_sort_idx[pitch_sort_idx]]
    active_idx = active_idx[time_sort_idx[pitch_sort_idx]]

    prev_note = -1

    # Iterate over the active indices
    notes = []
    for n, at in zip(pitch_idx, active_idx):

        # Create a new note if the pitch has changed
        if n != prev_note:
            prev_note = n
            # the notes are represented by a list containing
            # pitch, onset, offset and velocity.
            notes.append([n, at, at + 1, [pianoroll[n, at]]])

        # Otherwise update the offset of the note
        else:
            notes[-1][2] = at + 1
            notes[-1][3].append(pianoroll[n, at])

    # Create note array
    note_array = np.array([(p, float(on) / time_div,
                            (off - on) / time_div,
                            np.round(np.mean(vel)))
                           for p, on, off, vel in notes],
                          dtype=[('pitch', 'i4'),
                                 ('p_onset', 'f4'),
                                 ('p_duration', 'f4'),
                                 ('velocity', 'i4')])
    return note_array

def match_note_arrays(input_note_array, target_note_array,
                      array_type='performance', epsilon=0.01,
                      first_note_at_zero=False,
                      check_duration=True,
                      return_note_idxs=False):
    """Get indices of the notes from an input note array corresponding
    to a reference note array.

    Parameters
    ----------
    input_note_array : structured array
        Array containing performance/score information
    target_note_arr : structured array
        Array containing performance/score information, which which we
        want to match the input.
    array_type : 'performance' or 'score'
        Whether the structured arrays contain a performance or a score
    epsilon : float
        Epsilon for comparison of onset times.
    first_note_at_zero : bool
        If True, shifts the onsets of both note_arrays to start at 0.

    Returns
    -------
    matched_idxs : np.ndarray
        Indices of input_note_array corresponding to target_note_array

    Notes
    -----
    This is a greedy method. This method is useful to compare the
    *same performance* in different formats (e.g., in a match file and
    MIDI), or the *same score* (e.g., a MIDI file generated from a
    MusicXML file). It will not produce meaningful results between a
    score and a performance.

    """
    if array_type == 'performance':
        onset_key, duration_key = ('p_onset', 'p_duration')
    elif array_type == 'score':
        onset_key, duration_key = ('onset', 'duration')

    if first_note_at_zero:
        i_start = input_note_array[onset_key].min()
        t_start = target_note_array[onset_key].min()
    else:
        i_start, t_start = (0, 0)

    # sort indices
    i_sort_idx = np.argsort(input_note_array[onset_key])
    t_sort_idx = np.argsort(target_note_array[onset_key])

    # get onset, pitch and duration information
    i_onsets = input_note_array[onset_key][i_sort_idx] - i_start
    i_pitch = input_note_array['pitch'][i_sort_idx]

    t_onsets = target_note_array[onset_key][t_sort_idx] - t_start
    t_pitch = target_note_array['pitch'][t_sort_idx]

    if check_duration:
        i_duration = input_note_array[duration_key][i_sort_idx]
        t_duration = target_note_array[duration_key][t_sort_idx]

    matched_idxs = []
    matched_note_idxs = []
    count_double = 0
    for t, (i, o, p) in enumerate(zip(t_sort_idx, t_onsets, t_pitch)):
        # candidate onset idxs (between o - epsilon and o + epsilon)
        coix = np.where(np.logical_and(i_onsets >= o - epsilon,
                                       i_onsets <= o + epsilon))[0]
        if len(coix) > 0:
            # index of the note with the same pitch
            cpix = np.where(i_pitch[coix] == p)[0]
            if len(cpix) > 0:
                # index of the note with the closest duration
                if check_duration:
                    m_idx = abs(i_duration[coix[cpix]] - t_duration[t]).argmin()
                else:
                    m_idx = 0
                # match notes
                matched_idxs.append((int(i_sort_idx[coix[cpix[m_idx]]]), i))

    matched_idxs = np.array(matched_idxs)
    print("LENGTH OF MATCHED IDXS: ", len(matched_idxs),
          "LENGTH OF INPUT: ", len(input_note_array),
          "LENGTH OF TARGET: ", len(target_note_array))
    
    if return_note_idxs:
        if len(matched_idxs) > 0:
            matched_note_idxs = np.array([input_note_array["id"][matched_idxs[:,0]],
                                          target_note_array["id"][matched_idxs[:,1]]]).T
        return matched_idxs, matched_note_idxs
    else:
        return matched_idxs


if __name__ == '__main__':
    import doctest
    doctest.testmod()
