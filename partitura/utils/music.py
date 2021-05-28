#!/usr/bin/env python
from collections import defaultdict
import logging
import re

import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix

from partitura.utils.generic import find_nearest, search, iter_current_next

LOGGER = logging.getLogger(__name__)

MIDI_BASE_CLASS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
# _MORPHETIC_BASE_CLASS = {'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6}
# _MORPHETIC_OCTAVE = {0: 32, 1: 39, 2: 46, 3: 53, 4: 60, 5: 67, 6: 74, 7: 81, 8: 89}
ALTER_SIGNS = {None: "", 0: "", 1: "#", 2: "x", -1: "b", -2: "bb"}

DUMMY_PS_BASE_CLASS = {
    0: ("c", 0),
    1: ("c", 1),
    2: ("d", 0),
    3: ("d", 1),
    4: ("e", 0),
    5: ("f", 0),
    6: ("f", 1),
    7: ("g", 0),
    8: ("g", 1),
    9: ("a", 0),
    10: ("a", 1),
    11: ("b", 0),
}

LABEL_DURS = {
    "long": 16,
    "breve": 8,
    "whole": 4,
    "half": 2,
    "h": 2,
    "quarter": 1,
    "q": 1,
    "eighth": 1 / 2,
    "e": 1 / 2,
    "16th": 1 / 4,
    "32nd": 1 / 8.0,
    "64th": 1 / 16,
    "128th": 1 / 32,
    "256th": 1 / 64,
}
DOT_MULTIPLIERS = (1, 1 + 1 / 2, 1 + 3 / 4, 1 + 7 / 8)
# DURS and SYM_DURS encode the same information as _LABEL_DURS and
# _DOT_MULTIPLIERS, but they allow for faster estimation of symbolic duration
# (estimate_symbolic duration). At some point we will probably do away with
# _LABEL_DURS and _DOT_MULTIPLIERS.
DURS = np.array(
    [
        1.5625000e-02,
        2.3437500e-02,
        2.7343750e-02,
        2.9296875e-02,
        3.1250000e-02,
        4.6875000e-02,
        5.4687500e-02,
        5.8593750e-02,
        6.2500000e-02,
        9.3750000e-02,
        1.0937500e-01,
        1.1718750e-01,
        1.2500000e-01,
        1.8750000e-01,
        2.1875000e-01,
        2.3437500e-01,
        2.5000000e-01,
        3.7500000e-01,
        4.3750000e-01,
        4.6875000e-01,
        5.0000000e-01,
        5.0000000e-01,
        7.5000000e-01,
        7.5000000e-01,
        8.7500000e-01,
        8.7500000e-01,
        9.3750000e-01,
        9.3750000e-01,
        1.0000000e00,
        1.0000000e00,
        1.5000000e00,
        1.5000000e00,
        1.7500000e00,
        1.7500000e00,
        1.8750000e00,
        1.8750000e00,
        2.0000000e00,
        2.0000000e00,
        3.0000000e00,
        3.0000000e00,
        3.5000000e00,
        3.5000000e00,
        3.7500000e00,
        3.7500000e00,
        4.0000000e00,
        6.0000000e00,
        7.0000000e00,
        7.5000000e00,
        8.0000000e00,
        1.2000000e01,
        1.4000000e01,
        1.5000000e01,
        1.6000000e01,
        2.4000000e01,
        2.8000000e01,
        3.0000000e01,
    ]
)

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
    {"type": "long", "dots": 3},
]

MAJOR_KEYS = [
    "Cb",
    "Gb",
    "Db",
    "Ab",
    "Eb",
    "Bb",
    "F",
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
]
MINOR_KEYS = [
    "Ab",
    "Eb",
    "Bb",
    "F",
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
    "G#",
    "D#",
    "A#",
]

TIME_UNITS = ["beat", "quarter", "sec", "div"]

NOTE_NAME_PATT = re.compile(r"([A-G]{1})([xb\#]*)(\d+)")


def ensure_notearray(notearray_or_part, *args, **kwargs):
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
            raise ValueError("Input array is not a structured array!")

    elif isinstance(notearray_or_part, Part):
        return note_array_from_part(notearray_or_part, *args, **kwargs)

    elif isinstance(notearray_or_part, PartGroup):
        return note_array_from_part_list(notearray_or_part.children, *args, **kwargs)

    elif isinstance(notearray_or_part, PerformedPart):
        return notearray_or_part.note_array

    elif isinstance(notearray_or_part, list):
        if all([isinstance(part, Part) for part in notearray_or_part]):
            return note_array_from_part_list(notearray_or_part, *args, **kwargs)
        else:
            raise ValueError(
                "`notearray_or_part` should be a list of "
                "`Part` objects, but was given "
                "[{0}]".format(",".join(str(type(p)) for p in notearray_or_part))
            )
    else:
        raise ValueError(
            "`notearray_or_part` should be a structured "
            "numpy array, a `Part`, `PartGroup`, a "
            "`PerformedPart`, or a list but "
            "is {0}".format(type(notearray_or_part))
        )


def get_time_units_from_note_array(note_array):
    fields = set(note_array.dtype.fields)

    if fields is None:
        raise ValueError("`note_array` must be a structured numpy array")

    score_units = set(("onset_beat", "onset_quarter", "onset_div"))
    performance_units = set(("onset_sec",))

    if len(score_units.intersection(fields)) > 0:
        if "onset_beat" in fields:
            return ("onset_beat", "duration_beat")
        elif "onset_quarter" in fields:
            return ("onset_quarter", "duration_quarter")
        elif "onset_div" in fields:
            return ("onset_div", "duration_div")
    elif len(performance_units.intersection(fields)) > 0:
        return ("onset_sec", "duration_sec")
    else:
        raise ValueError("Input array does not contain the expected " "time-units")


def pitch_spelling_to_midi_pitch(step, alter, octave):
    midi_pitch = (octave + 1) * 12 + MIDI_BASE_CLASS[step.lower()] + (alter or 0)
    return midi_pitch


def midi_pitch_to_pitch_spelling(midi_pitch):
    octave = midi_pitch // 12 - 1
    step, alter = DUMMY_PS_BASE_CLASS[np.mod(midi_pitch, 12)]
    return ensure_pitch_spelling_format(step, alter, octave)


def note_name_to_pitch_spelling(note_name):
    note_info = NOTE_NAME_PATT.search(note_name)

    if note_info is None:
        raise ValueError("Invalid note name. "
                         "The note name must be "
                         "'<pitch class>(alteration)<octave>', "
                         f"but was given {note_name}.")
    step, alter, octave = note_info.groups()
    step, alter, octave = ensure_pitch_spelling_format(
        step=step,
        alter=alter if alter != "" else "n",
        octave=int(octave))
    return step, alter, octave


def note_name_to_midi_pitch(note_name):
    step, alter, octave = note_name_to_pitch_spelling(note_name)
    return pitch_spelling_to_midi_pitch(step, alter, octave)


def pitch_spelling_to_note_name(step, alter, octave):
    f_alter = ""
    if alter > 0:
        if alter == 2:
            f_alter = "x"
        else:
            f_alter = alter * "#"
    elif alter < 0:
        f_alter = abs(alter) * "b"

    note_name = f"{step.upper()}{f_alter}{octave}"
    return note_name


SIGN_TO_ALTER = {
    "n": 0,
    "#": 1,
    "x": 2,
    "##": 2,
    "###": 3,
    "b": -1,
    "bb": -2,
    "bbb": -3,
    "-": None,
}


def ensure_pitch_spelling_format(step, alter, octave):

    if step.lower() not in MIDI_BASE_CLASS:
        if step.lower() != "r":
            raise ValueError("Invalid `step`")

    if isinstance(alter, str):
        try:
            alter = SIGN_TO_ALTER[alter]
        except KeyError:
            raise ValueError(
                'Invalid `alter`, must be ("n", "#", '
                '"x", "b" or "bb"), but given {0}'.format(alter)
            )

    if not isinstance(alter, int):
        try:
            alter = int(alter)
        except TypeError or ValueError:
            if alter is not None:
                raise ValueError("`alter` must be an integer or None")

    if octave == "-":
        # check octave for weird rests in Batik match files
        octave = None
    else:
        if not isinstance(octave, int):
            try:
                octave = int(octave)
            except TypeError or ValueError:
                if octave is not None:
                    raise ValueError("`octave` must be an integer or None")

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

    if mode in ("minor", -1):
        keylist = MINOR_KEYS
        suffix = "m"
    elif mode in ("major", None, 1):
        keylist = MAJOR_KEYS
        suffix = ""
    else:
        raise Exception("Unknown mode {}".format(mode))

    try:
        name = keylist[fifths + 7]
    except IndexError:
        raise Exception("Unknown number of fifths {}".format(fifths))

    return name + suffix


def key_name_to_fifths_mode(name):
    """Return the number of sharps or flats and the mode of a key
    signature name. A negative number denotes the number of flats
    (i.e. -3 means three flats), and a positive number the number of
    sharps. The mode is specified as 'major' or 'minor'.

    Parameters
    ----------
    name : {"A", "A#m", "Ab", "Abm", "Am", "B", "Bb", "Bbm", "Bm", "C",\
"C#", "C#m", "Cb", "Cm", "D", "D#m", "Db", "Dm", "E", "Eb",\
"Ebm", "Em", "F", "F#", "F#m", "Fm", "G", "G#m", "Gb", "Gm"}
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

    if name.endswith("m"):
        mode = "minor"
        keylist = MINOR_KEYS
    else:
        mode = "major"
        keylist = MAJOR_KEYS

    try:
        fifths = keylist.index(name.strip("m")) - 7
    except ValueError:
        raise Exception("Unknown key signature {}".format(name))

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
    if mode in ("minor", -1):
        return -1
    elif mode in ("major", None, 1):
        return 1
    else:
        raise ValueError("Unknown mode {}".format(mode))


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
    if mode in ("minor", -1):
        return "minor"
    elif mode in ("major", None, 1):
        return "major"
    else:
        raise ValueError("Unknown mode {}".format(mode))


def estimate_symbolic_duration(dur, div, eps=10 ** -3):
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
    dots = unit.count(".")
    unit = unit.strip().rstrip(".")
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

        return "unknown"

    else:
        result = (symbolic_dur.get("type") or "") + "." * symbolic_dur.get("dots", 0)

        if "actual_notes" in symbolic_dur and "normal_notes" in symbolic_dur:

            result += "_{}/{}".format(
                symbolic_dur["actual_notes"], symbolic_dur["normal_notes"]
            )

        return result


def symbolic_to_numeric_duration(symbolic_dur, divs):
    numdur = divs * LABEL_DURS[symbolic_dur.get("type", None)]
    numdur *= DOT_MULTIPLIERS[symbolic_dur.get("dots", 0)]
    numdur *= (symbolic_dur.get("normal_notes") or 1) / (
        symbolic_dur.get("actual_notes") or 1
    )
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

    # gegeven b, kies alle veelvouden van 2*b, verschoven om b,
    # die tussen start en end liggen
    # gegeven b, kies alle veelvouden van 2*b die tussen start-b
    # en end-b liggen en tel er b bij op

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

    >>> find_tie_split(1, 8, 2)
    [(1, 8, {'type': 'half', 'dots': 2})]

    >>> find_tie_split(0, 3615, 480) # doctest: +NORMALIZE_WHITESPACE
    [(0, 3600, {'type': 'whole', 'dots': 3}),
     (3600, 3615, {'type': '128th', 'dots': 0})]

    """

    smallest_unit = find_smallest_unit(divs)

    def success(state):
        return all(
            estimate_symbolic_duration(right - left, divs)
            for left, right in iter_current_next([start] + state + [end])
        )

    def expand(state):
        if len(state) >= max_splits:
            return []
        else:
            split_start = ([start] + state)[-1]
            ordered_splits = order_splits(split_start, end, smallest_unit)
            new_states = [state + [s.item()] for s in ordered_splits]
            # start and end must be "in sync" with splits for states to succeed
            new_states = [
                s
                for s in new_states
                if (s[0] - start) % smallest_unit == 0
                and (end - s[-1]) % smallest_unit == 0
            ]
            return new_states

    def combine(new_states, old_states):
        return old_states + new_states

    states = [[]]

    # splits = search_recursive(states, success, expand, combine)
    splits = search(states, success, expand, combine)

    if splits is not None:
        solution = [
            (left, right, estimate_symbolic_duration(right - left, divs))
            for left, right in iter_current_next([start] + splits + [end])
        ]
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
    clefs = [
        dict(sign="F", line=4, octave_change=0),
        dict(sign="G", line=2, octave_change=0),
    ]
    f = interp1d([0, 49, 70, 127], [0, 0, 1, 1], kind="nearest")
    return clefs[int(f(center))]


def compute_pianoroll(
    note_info,
    time_unit="auto",
    time_div="auto",
    onset_only=False,
    note_separation=False,
    pitch_margin=-1,
    time_margin=0,
    return_idxs=False,
    piano_range=False,
    remove_drums=True,
):
    """Computes a piano roll from a structured note array (as
    generated by the `note_array` methods in `partitura.score.Part`
    and `partitura.performance.PerformedPart` instances).

    Parameters
    ----------
    note_info : structured array, `Part`, `PartGroup`, `PerformedPart`
        Note information
    time_unit : ('auto', 'beat', 'quarter', 'div', 'second')
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
    piano_range : bool, optional
        If True, the pitch axis of the piano roll is in piano keys
        instead of MIDI note numbers (and there are only 88 pitches).
        This is equivalent as slicing `piano_range_pianoroll =
        pianoroll[21:109, :]`.
    remove_drums: bool, optional
        If True, removes the drum track (i.e., channel 9) from the
        notes to be considered in the piano roll. This option is only
        relevant for piano rolls generated from a `PerformedPart`.
        Default is True.

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
    >>> from partitura.utils import compute_pianoroll
    >>> note_array = np.array([(60, 0, 1)],\
                          dtype=[('pitch', 'i4'),\
                                 ('onset_beat', 'f4'),\
                                 ('duration_beat', 'f4')])
    >>> pr = compute_pianoroll(note_array, pitch_margin=2, time_div=2)
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
    note_array = ensure_notearray(note_info)

    if time_unit not in TIME_UNITS + ["auto"]:
        raise ValueError(
            "`time_unit` must be one of "
            '{0} or "auto", but was given '
            "{1}".format(", ".join(TIME_UNITS), time_unit)
        )
    if time_unit == "auto":
        onset_unit, duration_unit = get_time_units_from_note_array(note_array)

    else:
        onset_unit = f"onset_{time_unit}"
        duration_unit = f"duration_{time_unit}"

    if time_div == "auto":
        if onset_unit in ("onset_beat", "onset_quarter", "onset_sec"):
            time_div = 8
        elif onset_unit == "onset_div":
            time_div = 1
    else:
        time_div = int(time_div)

    if "channel" in note_array.dtype.names and remove_drums:
        LOGGER.info("Do not consider drum track for computing piano roll")
        non_drum_idxs = np.where(note_array["channel"] != 9)[0]
        note_array = note_array[non_drum_idxs]

    piano_roll_fields = ["pitch", onset_unit, duration_unit]

    if "velocity" in note_array.dtype.names:
        piano_roll_fields += ["velocity"]

    pr_input = np.column_stack(
        [note_array[field].astype(float) for field in piano_roll_fields]
    )

    return _make_pianoroll(
        note_info=pr_input,
        time_div=time_div,
        onset_only=onset_only,
        note_separation=note_separation,
        pitch_margin=pitch_margin,
        time_margin=time_margin,
        return_idxs=return_idxs,
        piano_range=piano_range,
    )


def _make_pianoroll(
    note_info,
    onset_only=False,
    pitch_margin=-1,
    time_margin=0,
    time_div=8,
    note_separation=True,
    return_idxs=False,
    piano_range=False,
    remove_silence=True,
    min_time=None,
):
    # non-public
    """Computes a piano roll from a numpy array with MIDI pitch,
    onset, duration and (optionally) MIDI velocity information. See
    `compute_pianoroll` for a complete description of the
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
    if min_time is None:
        min_time = 0 if min(onset) >= 0 else min(onset)
        if remove_silence:
            min_time = onset[0]
    else:
        if min_time > min(onset):
            raise ValueError(
                "`min_time` must be smaller or equal than " "the smallest onset time "
            )
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
    pr_onset = np.round(time_div * onset).astype(int)
    pr_offset = np.round(time_div * offset).astype(int)

    # Determine the non-zero indices of the piano roll
    if onset_only:
        _idx_fill = np.column_stack([pr_pitch, pr_onset, pr_velocity])
    else:
        pr_offset = np.maximum(pr_onset + 1, pr_offset - (1 if note_separation else 0))
        _idx_fill = np.vstack(
            [
                np.column_stack(
                    (
                        np.zeros(off - on) + pitch,
                        np.arange(on, off),
                        np.zeros(off - on) + vel,
                    )
                )
                for on, off, pitch, vel in zip(
                    pr_onset, pr_offset, pr_pitch, pr_velocity
                )
                if off <= N
            ]
        )

    # Fix multiple notes with the same pitch and onset
    fill_dict = defaultdict(list)
    for row, col, vel in _idx_fill:
        key = (int(row), int(col))
        fill_dict[key].append(vel)

    idx_fill = np.zeros((len(fill_dict), 3))
    for i, ((row, column), vel) in enumerate(fill_dict.items()):
        idx_fill[i] = np.array([row, column, max(vel)])

    # Fill piano roll
    pianoroll = csc_matrix(
        (idx_fill[:, 2], (idx_fill[:, 0], idx_fill[:, 1])), shape=(M, N), dtype=int
    )

    pr_idx_pitch_start = 0
    if piano_range:
        pianoroll = pianoroll[21:109, :]
        pr_idx_pitch_start = 21

    if return_idxs:
        # indices of each note in the piano roll
        pr_idx = np.column_stack(
            [pr_pitch - pr_idx_pitch_start, pr_onset, pr_offset]
        ).astype(int)
        return pianoroll, pr_idx[idx.argsort()]
    else:
        return pianoroll


def pianoroll_to_notearray(pianoroll, time_div=8, time_unit="sec"):
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
    time_unit : {'beat', 'quarter', 'div', 'sec'}
        time unit of the output note array.

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
    pitch_sort_idx = pitch_idx[time_sort_idx].argsort(kind="mergesort")

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
    note_array = np.array(
        [
            (p, float(on) / time_div, (off - on) / time_div, np.round(np.mean(vel)))
            for p, on, off, vel in notes
        ],
        dtype=[
            ("pitch", "i4"),
            (f"onset_{time_unit}", "f4"),
            (f"duration_{time_unit}", "f4"),
            ("velocity", "i4"),
        ],
    )
    return note_array


def match_note_arrays(
    input_note_array,
    target_note_array,
    fields=None,
    epsilon=0.01,
    first_note_at_zero=False,
    check_duration=True,
    return_note_idxs=False,
):
    """Compute a greedy matching of the notes of two note_arrays based on
    onset, pitch and (optionally) duration. Returns an array of matched
    note_array indices and (optionally) an array of the corresponding
    matched note indices.

    Get an array of note_array indices of the notes from an input note
    array corresponding to a reference note array.

    Parameters
    ----------
    input_note_array : structured array
        Array containing performance/score information
    target_note_arr : structured array
        Array containing performance/score information, which which we
        want to match the input.
    fields : strings or tuple of strings
        Field names to use for onset and duration in note_arrays.
        If None defaults to beats or seconds, respectively.
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
    *same performance* in different formats or versions (e.g., in a
    match file and MIDI), or the *same score* (e.g., a MIDI file generated
    from a MusicXML file). It will not produce meaningful results between a
    score and a performance.

    """
    input_note_array = ensure_notearray(input_note_array)
    target_note_array = ensure_notearray(target_note_array)

    if fields is not None:

        if isinstance(fields, (list, tuple)):
            onset_key, duration_key = fields
        elif isinstance(fields, str):
            onset_key = fields
            duration_key = None

            if duration_key is None and check_duration:
                check_duration = False
        else:
            raise ValueError("`fields` should be a tuple or a string, but given "
                             f"{type(fields)}")
    else:
        onset_key, duration_key = get_time_units_from_note_array(input_note_array)
        onset_key_check, _ = get_time_units_from_note_array(target_note_array)
        if onset_key_check != onset_key:
            raise ValueError("Input and target arrays have different field names!")

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
    i_pitch = input_note_array["pitch"][i_sort_idx]

    t_onsets = target_note_array[onset_key][t_sort_idx] - t_start
    t_pitch = target_note_array["pitch"][t_sort_idx]

    if check_duration:
        i_duration = input_note_array[duration_key][i_sort_idx]
        t_duration = target_note_array[duration_key][t_sort_idx]

    matched_idxs = []
    matched_note_idxs = []

    # dictionary of lists. For each index of the target, get a list of the
    # corresponding indices in the input
    matched_target = defaultdict(list)
    # dictionary of lists. For each index of the input, get a list of the
    # corresponding indices in the target
    matched_input = defaultdict(list)
    for t, (i, o, p) in enumerate(zip(t_sort_idx, t_onsets, t_pitch)):
        # candidate onset idxs (between o - epsilon and o + epsilon)
        coix = np.where(
            np.logical_and(i_onsets >= o - epsilon, i_onsets <= o + epsilon)
        )[0]
        if len(coix) > 0:
            # index of the note with the same pitch
            cpix = np.where(i_pitch[coix] == p)[0]
            if len(cpix) > 0:
                m_idx = 0
                # index of the note with the closest duration
                if len(cpix) > 1 and check_duration:
                    m_idx = abs(i_duration[coix[cpix]] - t_duration[t]).argmin()
                # match notes
                input_idx = int(i_sort_idx[coix[cpix[m_idx]]])
                target_idx = i
                # matched_idxs.append((input_idx, target_idx))
                matched_input[input_idx].append(target_idx)
                matched_target[target_idx].append(input_idx)

    matched_target_idxs = []
    for inix, taix in matched_input.items():
        if len(taix) > 1:
            # For the case that there are multiple notes aligned to the input note

            # get indices of the target notes if they have not yet been used
            taix_to_consider = np.array([ti for ti in taix
                                         if ti not in matched_target_idxs],
                                        dtype=int)
            if len(taix_to_consider) > 0:
                # If there are some indices to consider
                candidate_notes = target_note_array[taix_to_consider]

                if check_duration:
                    best_candidate_idx = \
                        (candidate_notes[duration_key] -
                         input_note_array[inix][duration_key]).argmin()
                else:
                    # Take the first one if no other information is given
                    best_candidate_idx = 0

                matched_idxs.append((inix, taix_to_consider[best_candidate_idx]))
                matched_target_idxs.append(taix_to_consider[best_candidate_idx])
        else:
            matched_idxs.append((inix, taix[0]))
            matched_target_idxs.append(taix[0])
    matched_idxs = np.array(matched_idxs)

    LOGGER.info("Length of matched idxs: " "{0}".format(len(matched_idxs)))
    LOGGER.info("Length of input note_array: " "{0}".format(len(input_note_array)))
    LOGGER.info("Length of target note_array: " "{0}".format(len(target_note_array)))

    if return_note_idxs:
        if len(matched_idxs) > 0:
            matched_note_idxs = np.array(
                [
                    input_note_array["id"][matched_idxs[:, 0]],
                    target_note_array["id"][matched_idxs[:, 1]],
                ]
            ).T
        return matched_idxs, matched_note_idxs
    else:
        return matched_idxs


def remove_silence_from_performed_part(ppart):
    """
    Remove silence at the beginning of a PerformedPart
    by shifting notes, controls and programs to the beginning
    of the file.

    Parameters
    ----------
    ppart : `PerformedPart`
        A performed part. This part will be edited in-place.
    """
    # Consider only Controls and Notes, since by default,
    # programs are created at the beginning of the file.
    # c_times = [c['time'] for c in ppart.controls]
    n_times = [n["note_on"] for n in ppart.notes]
    start_time = min(n_times)

    shifted_controls = []
    control_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for c in ppart.controls:
        control_dict[c["track"]][c["channel"]][c["number"]].append(
            (c["time"], c["value"])
        )

    for track in control_dict:
        for channel in control_dict[track]:
            for number, ct in control_dict[track][channel].items():

                cta = np.array(ct)
                cinterp = interp1d(
                    x=cta[:, 0],
                    y=cta[:, 1],
                    kind="previous",
                    bounds_error=False,
                    fill_value=(cta[0, 1], cta[-1, 1]),
                )

                c_idxs = np.where(cta[:, 0] >= start_time)[0]

                c_times = cta[c_idxs, 0]
                if start_time not in c_times:
                    c_times = np.r_[start_time, c_times]

                c_values = cinterp(c_times)

                for t, v in zip(c_times, c_values):
                    shifted_controls.append(
                        dict(
                            time=max(t - start_time, 0),
                            number=number,
                            value=int(v),
                            track=track,
                            channel=channel,
                        )
                    )

    # sort controls according to time
    shifted_controls.sort(key=lambda x: x["time"])

    ppart.controls = shifted_controls

    # Shift notes
    for note in ppart.notes:
        note["note_on"] = max(note["note_on"] - start_time, 0)
        note["note_off"] = max(note["note_off"] - start_time, 0)
        note["sound_off"] = max(note["sound_off"] - start_time, 0)

    # Shift programs
    for program in ppart.programs:
        program["time"] = max(program["time"] - start_time, 0)


def note_array_from_part_list(
    part_list,
    unique_id_per_part=True,
    include_pitch_spelling=False,
    include_key_signature=False,
    include_time_signature=False,
):
    """
    Construct a structured Note array from a list of Part objects

    Parameters
    ----------
    part_list : list
       A list of `Part` or `PerformedPart` objects. All elements in
       the list must be of the same type (i.e., no mixing `Part`
       and `PerformedPart` objects in the same list.
    unique_id_per_part : bool (optional)
       Indicate from which part do each note come from in the note ids.
    include_pitch_spelling: bool (optional)
       Include pitch spelling information in note array. Only valid
       if parts in `part_list` are `Part` objects. See `note_array_from_part`
       for more info. Default is False.
    include_key_signature: bool (optional)
       Include key signature information in output note array.
       Only valid if parts in `part_list` are `Part` objects.
       See `note_array_from_part` for more info. Default is False.
    include_time_signature : bool (optional)
       Include time signature information in output note array.
       Only valid if parts in `part_list` are `Part` objects.
       See `note_array_from_part` for more info. Default is False.

    Returns
    -------
    note_array: structured array
        A structured array containing pitch, onset, duration, voice
        and id for each note in each part of the `part_list`. The note
        ids in this array include the number of the part to which they
        belong.
    """
    from partitura.score import Part, PartGroup
    from partitura.performance import PerformedPart

    note_array = []
    for i, part in enumerate(part_list):
        if isinstance(part, (Part, PartGroup)):
            if isinstance(part, Part):
                na = note_array_from_part(
                    part=part,
                    include_pitch_spelling=include_pitch_spelling,
                    include_key_signature=include_key_signature,
                    include_time_signature=include_time_signature,
                )
            elif isinstance(part, PartGroup):
                na = note_array_from_part_list(
                    part_list=part.children,
                    unique_id_per_part=unique_id_per_part,
                    include_pitch_spelling=include_pitch_spelling,
                    include_key_signature=include_key_signature,
                    include_time_signature=include_time_signature,
                )
        elif isinstance(part, PerformedPart):
            na = part.note_array
        if unique_id_per_part:
            # Update id with part number
            na["id"] = np.array(
                ["P{0:02d}_".format(i) + nid for nid in na["id"]], dtype=na["id"].dtype
            )
        note_array.append(na)

    # concatenate note_arrays
    note_array = np.hstack(note_array)

    onset_unit, _ = get_time_units_from_note_array(note_array)

    # sort by onset and pitch
    pitch_sort_idx = np.argsort(note_array["pitch"])
    note_array = note_array[pitch_sort_idx]
    onset_sort_idx = np.argsort(note_array[onset_unit], kind="mergesort")
    note_array = note_array[onset_sort_idx]

    return note_array


def slice_notearray_by_time(
    note_array, start_time, end_time, time_unit="auto", clip_onset_duration=True
):
    """
    Get a slice of a structured note array by time

    Parameters
    ----------
    note_array : structured array
        Structured array with score information.
    start_time : float
        Starting time
    end_time : float
        End time
    time_unit : {'auto', 'beat', 'quarter', 'second', 'div'} optional
        Time unit. If 'auto', the default time unit will be inferred
        from the note_array.
    clip_onset_duration : bool optional
        Clip duration of the notes in the array to fit within the
        specified window

    Returns
    -------
    note_array_slice : stuctured array
        Structured array with only the score information between
        `start_time` and `end_time`.

    TODO
    ----
    * adjust onsets and duration in other units
    """

    if time_unit not in TIME_UNITS + ["auto"]:
        raise ValueError(
            "`time_unit` must be 'beat', 'quarter', "
            "'sec', 'div' or 'auto', but is "
            "{0}".format(time_unit)
        )
    if time_unit == "auto":
        onset_unit, duration_unit = get_time_units_from_note_array(note_array)
    else:
        onset_unit, duration_unit = [
            "{0}_{1}".format(d, time_unit) for d in ("onset", "duration")
        ]

    onsets = note_array[onset_unit]
    offsets = note_array[onset_unit] + note_array[duration_unit]

    starting_idxs = set(np.where(onsets >= start_time)[0])
    ending_idxs = set(np.where(onsets < end_time)[0])

    prev_starting_idxs = set(np.where(onsets < start_time)[0])
    sounding_after_start_idxs = set(np.where(offsets > start_time)[0])

    active_idx = np.array(
        list(
            starting_idxs.intersection(ending_idxs).union(
                prev_starting_idxs.intersection(sounding_after_start_idxs)
            )
        )
    )
    active_idx.sort()

    if len(active_idx) == 0:
        # If there are no elements, return an empty array
        note_array_slice = np.empty(0, dtype=note_array.dtype)
    else:
        note_array_slice = note_array[active_idx]

    if clip_onset_duration and len(active_idx) > 0:
        psi = np.where(note_array_slice[onset_unit] < start_time)[0]
        note_array_slice[psi] = start_time
        adj_offsets = np.clip(
            note_array_slice[onset_unit] + note_array_slice[duration_unit],
            a_min=None,
            a_max=end_time,
        )
        note_array_slice[duration_unit] = adj_offsets - note_array_slice[onset_unit]

    return note_array_slice


def note_array_from_part(
    part,
    include_pitch_spelling=False,
    include_key_signature=False,
    include_time_signature=False,
):
    """
    Create a structured array with note information
    from a `Part` object.

    Parameters
    ----------
    part : partitura.score.Part
        An object representing a score part.
    include_pitch_spelling : bool (optional)
        If `True`, includes pitch spelling information for each
        note. Default is False
    include_key_signature : bool (optional)
        If `True`, includes key signature information, i.e.,
        the key signature at the onset time of each note (all
        notes starting at the same time have the same key signature).
        Default is False
    include_time_signature : bool (optional)
        If `True`,  includes time signature information, i.e.,
        the time signature at the onset time of each note (all
        notes starting at the same time have the same time signature).
        Default is False

    Returns
    -------
    note_array : structured array
        A structured array containing note information. The fields are
            * 'onset_beat': onset time of the note in beats
            * 'duration_beat': duration of the note in beats
            * 'onset_quarter': onset time of the note in quarters
            * 'duration_quarter': duration of the note in quarters
            * 'onset_div': onset of the note in divs (e.g., MIDI ticks,
              divisions in MusicXML)
            * 'duration_div': duration of the note in divs
            * 'pitch': MIDI pitch of a note.
            * 'voice': Voice number of a note (if given in the score)
            * 'id': Id of the note

        If `include_pitch_spelling` is True:
            * 'step': name of the note ("C", "D", "E", "F", "G", "A", "B")
            * 'alter': alteration (0=natural, -1=flat, 1=sharp,
              2=double sharp, etc.)
            * 'octave': octave of the note.

        If `include_key_signature` is True:
            * 'ks_fifths': Fifths starting from C in the circle of fifths
            * 'mode': major or minor

        If `include_time_signature` is True:
            * 'ts_beats': number of beats in a measure
            * 'ts_beat_type': type of beats (denominator of the time signature)

    Examples
    --------
    >>> from partitura import load_musicxml, EXAMPLE_MUSICXML
    >>> from partitura.utils import note_array_from_part
    >>> part = load_musicxml(EXAMPLE_MUSICXML)
    >>> note_array_from_part(part, True, True, True) # doctest: +NORMALIZE_WHITESPACE
    array([(0., 4., 0., 4.,  0, 48, 69, 1, 'n01', 'A', 0, 4, 0, 1, 4, 4),
           (2., 2., 2., 2., 24, 24, 72, 2, 'n02', 'C', 0, 5, 0, 1, 4, 4),
           (2., 2., 2., 2., 24, 24, 76, 2, 'n03', 'E', 0, 5, 0, 1, 4, 4)],
          dtype=[('onset_beat', '<f4'),
                 ('duration_beat', '<f4'),
                 ('onset_quarter', '<f4'),
                 ('duration_quarter', '<f4'),
                 ('onset_div', '<i4'),
                 ('duration_div', '<i4'),
                 ('pitch', '<i4'),
                 ('voice', '<i4'),
                 ('id', '<U256'),
                 ('step', '<U256'),
                 ('alter', '<i4'),
                 ('octave', '<i4'),
                 ('ks_fifths', '<i4'),
                 ('ks_mode', '<i4'),
                 ('ts_beats', '<i4'),
                 ('ts_beat_type', '<i4')])
    """
    if include_time_signature:
        time_signature_map = part.time_signature_map
    else:
        time_signature_map = None

    if include_key_signature:
        key_signature_map = part.key_signature_map
    else:
        key_signature_map = None

    note_array = note_array_from_note_list(
        note_list=part.notes_tied,
        beat_map=part.beat_map,
        quarter_map=part.quarter_map,
        time_signature_map=time_signature_map,
        key_signature_map=key_signature_map,
        include_pitch_spelling=include_pitch_spelling)
    return note_array


def note_array_from_note_list(
        note_list,
        beat_map=None,
        quarter_map=None,
        time_signature_map=None,
        key_signature_map=None,
        include_pitch_spelling=False,
):
    """
    Create a structured array with note information
    from a a list of `Note` objects.

    Parameters
    ----------
    note_list : list of `Note` objects
        A list of `Note` objects containing score information.
    beat_map : callable or None
        A function that maps score time in divs to score time in beats.
        If `None` is given, the output structured array will not
        include this information.
    quarter_map: callable or None
        A function that maps score time in divs to score time in quarters.
        If `None` is given, the output structured array will not
        include this information.
    time_signature_map: callable or None (optional)
        A function that maps score time in divs to the time signature at
        that time (in terms of number of beats and beat type).
        If `None` is given, the output structured array will not
        include this information.
    key_signature_map: callable or None (optional)
        A function that maps score time in divs to the key signature at
        that time (in terms of fifths and mode).
        If `None` is given, the output structured array will not
        include this information.
    include_pitch_spelling : bool (optional)
        If `True`, includes pitch spelling information for each
        note. Default is False

    Returns
    -------
    note_array : structured array
        A structured array containing note information. The fields are
            * 'onset_beat': onset time of the note in beats.
              Included if `beat_map` is not `None`.
            * 'duration_beat': duration of the note in beats.
              Included if `beat_map` is not `None`.
            * 'onset_quarter': onset time of the note in quarters.
              Included if `quarter_map` is not `None`.
            * 'duration_quarter': duration of the note in quarters.
              Included if `quarter_map` is not `None`.
            * 'onset_div': onset of the note in divs (e.g., MIDI ticks,
              divisions in MusicXML)
            * 'duration_div': duration of the note in divs
            * 'pitch': MIDI pitch of a note.
            * 'voice': Voice number of a note (if given in the score)
            * 'id': Id of the note
            * 'step': name of the note ("C", "D", "E", "F", "G", "A", "B").
              Included if `include_pitch_spelling` is `True`.
            * 'alter': alteration (0=natural, -1=flat, 1=sharp,
              2=double sharp, etc.). Included if `include_pitch_spelling`
              is `True`.
            * 'octave': octave of the note. Included if `include_pitch_spelling`
              is `True`.
            * 'ks_fifths': Fifths starting from C in the circle of fifths.
              Included if `key_signature_map` is not `None`.
            * 'mode': major or minor. Included If `key_signature_map` is
              not `None`.
            * 'ts_beats': number of beats in a measure. If `time_signature_map`
               is True.
            * 'ts_beat_type': type of beats (denominator of the time signature).
              If `include_time_signature` is True.
    """

    fields = []
    if beat_map is not None:
        # Preserve the order of the fields
        fields += [("onset_beat", "f4"),
                   ("duration_beat", "f4")]

    if quarter_map is not None:
        fields += [("onset_quarter", "f4"),
                   ("duration_quarter", "f4")]
    fields += [
        ("onset_div", "i4"),
        ("duration_div", "i4"),
        ("pitch", "i4"),
        ("voice", "i4"),
        ("id", "U256"),
    ]

    # fields for pitch spelling
    if include_pitch_spelling:
        fields += [("step", "U256"), ("alter", "i4"), ("octave", "i4")]

    # fields for key signature
    if key_signature_map is not None:
        fields += [("ks_fifths", "i4"), ("ks_mode", "i4")]

    # fields for time signature
    if time_signature_map is not None:
        fields += [("ts_beats", "i4"), ("ts_beat_type", "i4")]

    note_array = []
    for note in note_list:

        note_info = tuple()
        note_on_div = note.start.t
        note_off_div = note.start.t + note.duration_tied
        note_dur_div = note_off_div - note_on_div

        if beat_map is not None:
            note_on_beat, note_off_beat = beat_map([note_on_div, note_off_div])
            note_dur_beat = note_off_beat - note_on_beat

            note_info += (note_on_beat,
                          note_dur_beat)

        if quarter_map is not None:
            note_on_quarter, note_off_quarter = quarter_map(
                [note_on_div, note_off_div]
            )
            note_dur_quarter = note_off_quarter - note_on_quarter

            note_info += (note_on_quarter,
                          note_dur_quarter)

        note_info += (
            note_on_div,
            note_dur_div,
            note.midi_pitch,
            note.voice if note.voice is not None else -1,
            note.id,
        )

        if include_pitch_spelling:
            step = note.step
            alter = note.alter if note.alter is not None else 0
            octave = note.octave

            note_info += (step, alter, octave)

        if key_signature_map is not None:
            fifths, mode = key_signature_map(note.start.t)

            note_info += (fifths, mode)

        if time_signature_map is not None:
            beats, beat_type = time_signature_map(note.start.t)

            note_info += (beats, beat_type)

        note_array.append(note_info)

    note_array = np.array(note_array, dtype=fields)

    # Sanitize voice information
    no_voice_idx = np.where(note_array["voice"] == -1)[0]
    max_voice = note_array["voice"].max()
    note_array["voice"][no_voice_idx] = max_voice + 1

    return note_array


def update_note_ids_after_unfolding(part):

    note_id_dict = defaultdict(list)

    for n in part.notes:
        note_id_dict[n.id].append(n)

    for nid, notes in note_id_dict.items():

        if nid is None:
            continue

        notes.sort(key=lambda x: x.start.t)

        for i, note in enumerate(notes):
            note.id = f"{note.id}-{i+1}"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
