#!/usr/bin/env python
from collections import defaultdict
import re
import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix
from typing import Union
from partitura.utils.generic import find_nearest, search, iter_current_next

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

MEI_DURS_TO_SYMBOLIC = {
    "long": "long",
    "0": "breve",
    "1": "whole",
    "2": "half",
    "4": "quarter",
    "8": "eighth",
    "16": "16th",
    "32": "32nd",
    "64": "64th",
    "128": "128th",
    "256": "256th",
}

SYMBOLIC_TO_INT_DURS = {
    "long": 0.25,
    "breve": 0.5,
    "whole": 1,
    "half": 2,
    "quarter": 4,
    "eighth": 8,
    "16th": 16,
    "32nd": 32,
    "64th": 64,
    "128th": 128,
    "256th": 256,
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

MUSICAL_BEATS = {6: 2, 9: 3, 12: 4}

# Standard tuning frequency of A4 in Hz
A4 = 440.0


def ensure_notearray(notearray_or_part, *args, **kwargs):
    """
    Ensures to get a structured note array from the input.

    Parameters
    ----------
    notearray_or_part : structured ndarray, `Score`, `Part`, `PerformedPart`
        Input score information
    kwargs : dict
        Additional arguments to be passed to `partitura.utils.note_array_from_part()`.

    Returns
    -------
    structured ndarray
        Structured array containing score information.
    """
    from partitura.score import Part, PartGroup, Score
    from partitura.performance import PerformedPart, Performance

    if isinstance(notearray_or_part, np.ndarray):
        if notearray_or_part.dtype.fields is not None:
            return notearray_or_part
        else:
            raise ValueError("Input array is not a structured array!")

    elif isinstance(notearray_or_part, Part):
        return note_array_from_part(notearray_or_part, *args, **kwargs)

    elif isinstance(notearray_or_part, PartGroup):
        return note_array_from_part_list(notearray_or_part.children, *args, **kwargs)

    elif isinstance(notearray_or_part, Score):
        return note_array_from_part_list(notearray_or_part.parts, *args, **kwargs)

    elif isinstance(notearray_or_part, (PerformedPart, Performance)):
        return notearray_or_part.note_array(*args, **kwargs)
    elif isinstance(notearray_or_part, Score):
        return notearray_or_part.note_array(*args, **kwargs)
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


def ensure_rest_array(restarray_or_part, *args, **kwargs):
    """
    Ensures to get a structured note array from the input.

    Parameters
    ----------
    restarray_or_part : structured ndarray, `Part` or `PerformedPart`
        Input score information

    Returns
    -------
    structured ndarray
        Structured array containing score information.
    """
    from partitura.score import Part, PartGroup

    if isinstance(restarray_or_part, np.ndarray):
        if restarray_or_part.dtype.fields is not None:
            return restarray_or_part
        else:
            raise ValueError("Input array is not a structured array!")

    elif isinstance(restarray_or_part, Part):
        return rest_array_from_part(restarray_or_part, *args, **kwargs)

    elif isinstance(restarray_or_part, PartGroup):
        return rest_array_from_part_list(restarray_or_part.children, *args, **kwargs)

    elif isinstance(restarray_or_part, list):
        if all([isinstance(part, Part) for part in restarray_or_part]):
            return rest_array_from_part_list(restarray_or_part, *args, **kwargs)
        else:
            raise ValueError(
                "`restarray_or_part` should be a list of "
                "`Part` objects, but was given "
                "[{0}]".format(",".join(str(type(p)) for p in restarray_or_part))
            )
    else:
        raise ValueError(
            "`restarray_or_part` should be a structured "
            "numpy array, a `Part`, `PartGroup`, or a list but "
            "is {0}".format(type(restarray_or_part))
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
        raise ValueError(
            "Invalid note name. "
            "The note name must be "
            "'<pitch class>(alteration)<octave>', "
            f"but was given {note_name}."
        )
    step, alter, octave = note_info.groups()
    step, alter, octave = ensure_pitch_spelling_format(
        step=step, alter=alter if alter != "" else "n", octave=int(octave)
    )
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


def midi_pitch_to_frequency(
        midi_pitch: Union[int, float, np.ndarray], a4: Union[int, float] = A4
) -> Union[float, np.ndarray]:
    """
    Convert MIDI pitch to frequency in Hz. This method assumes equal temperament.

    Parameters
    ----------
    midi_pitch: int, float or ndarray
        MIDI pitch of the note(s).
    a4 : int or float (optional)
        Frequency of A4 in Hz. By default is 440 Hz.

    Returns
    -------
    freq : float or ndarray
        Frequency of the note(s).
    """
    freq = (a4 / 32) * (2 ** ((midi_pitch - 9) / 12))
    return freq


def frequency_to_midi_pitch(
        freq: Union[int, float, np.ndarray],
        a4: Union[int, float] = A4,
) -> Union[int, np.ndarray]:
    """
    Convert frequency to MIDI pitch. This method assumes equal temperament.

    Parameters
    ----------
    freq : float, int or np.ndarray
        Frequency of the note(s) in Hz.
    a4 : int or float (optional)
        Frequency of A4 in Hz. By default is 440 Hz.

    Returns
    -------
    midi_pitch : int or np.ndarray
        MIDI pitch of the notes.
    """
    midi_pitch = np.round(12 * np.log2(32 * freq / a4) + 9)

    if isinstance(midi_pitch, (int, float)):
        return int(midi_pitch)
    elif isinstance(midi_pitch, np.ndarray):
        return midi_pitch.astype(int)


SIGN_TO_ALTER = {
    "n": 0,
    "#": 1,
    "s": 1,
    "ss": 2,
    "x": 2,
    "##": 2,
    "###": 3,
    "b": -1,
    "f": -1,
    "bb": -2,
    "ff": -2,
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


def key_name_to_fifths_mode(key_name):
    """Return the number of sharps or flats and the mode of a key
    signature name. A negative number denotes the number of flats
    (i.e. -3 means three flats), and a positive number the number of
    sharps. The mode is specified as 'major' or 'minor'.

    Parameters
    ----------
    name : str
        Name of the key signature, i.e. Am, E#, etc

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
    fifths_list = ["F", "C", "G", "D", "A", "E", "B"]

    if "m" in key_name:
        mode = "minor"
        s_list = fifths_list[4:] + fifths_list[:4]
        if "b" in key_name or (len(key_name) == 2 and s_list.index(key_name[0]) > 2):
            idx = s_list[::-1].index(key_name[0]) + 1
            corr = 1 if idx > 4 else 0
            fifths = -idx - 7 * (key_name.count("b") - corr)
        else:
            idx = s_list.index(key_name[0])
            corr = 1 if idx > 2 else 0
            fifths = idx + 7 * (key_name.count("#") - corr)
    else:
        mode = "major"
        s_list = fifths_list[1:] + fifths_list[:1]
        if "b" in key_name or key_name == "F":
            idx = s_list[::-1].index(key_name[0]) + 1
            corr = 1 if idx > 1 else 0
            fifths = -idx - 7 * (key_name.count("b") - corr)
        else:
            idx = s_list.index(key_name[0])
            corr = 1 if idx > 5 else 0
            fifths = idx + 7 * (key_name.count("#") - corr)
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
    remove_silence=True,
    end_time=None,
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
    remove_drums : bool, optional
        If True, removes the drum track (i.e., channel 9) from the
        notes to be considered in the piano roll. This option is only
        relevant for piano rolls generated from a `PerformedPart`.
        Default is True.
    remove_silence : bool, optional
        If True, the first frame of the pianoroll starts at the onset
        of the first note, not at time 0 of the timeline.
    end_time : int, optional
        The time corresponding to the ending of the last 
        pianoroll frame (in time_unit). 
        If None this is set to the last note offset.

    Returns
    -------
    pianoroll : scipy.sparse.csr_matrix
        A sparse int matrix of size representing the pianoroll; The
        first dimension is pitch, the second is time; The sizes of the
        dimensions vary with the parameters `pitch_margin`,
        `time_margin`, `time_div`, `remove silence`, and `end_time`.
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
        warnings.warn("Do not consider drum track for computing piano roll")
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
        remove_silence=remove_silence,
        end_time=end_time,
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
    end_time=None,
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
    duration = note_info[:, 2]

    if np.any(duration < 0):
        raise ValueError("Note durations should be >= 0!")

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
    duration = duration[idx]

    if min_time is None:
        min_time = 0 if min(onset) >= 0 else min(onset)
        if remove_silence:
            min_time = onset[0]
    else:
        if min_time > min(onset):
            raise ValueError(
                "`min_time` must be smaller or equal than " "the smallest onset time "
            )

    onset -= min_time
    if end_time is not None:
        end_time -= min_time

    if pitch_margin > -1:
        pr_pitch -= lowest_pitch
        pr_pitch += pitch_margin

    # Size of the output piano roll
    # Pitch dimension
    if pitch_margin > -1:
        M = int(pitch_span + 2 * pitch_margin)
    else:
        M = int(pitch_span)

    # Onset and offset times of the notes in the piano roll
    pr_onset = np.round(time_div * onset).astype(int)
    pr_onset += int(time_margin * time_div)
    pr_duration = np.clip(
        np.round(time_div * duration).astype(int), a_max=None, a_min=1
    )
    pr_offset = pr_onset + pr_duration

    # Time dimension
    if end_time is None:
        N = int(np.ceil(time_div * time_margin + pr_offset.max()))
    else:
        if end_time * time_div < pr_offset.max():
            raise ValueError(
                "`end_time` must be higher or equal than the last note offset time"
            )
        else:
            N = int(np.ceil(time_div * time_margin + time_div * end_time))

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
        Structured array with pitch, onset, duration, velocity
        and note id fields.

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
    # check size of the piano roll
    init_pitch = 0
    if pianoroll.shape[0] != 128:
        if pianoroll.shape[0] == 88:
            init_pitch = 21
        else:
            raise ValueError(
                "The shape of the piano roll must be (128, n_time_steps) or"
                f"(88, n_timesteps) but is {pianoroll.shape}"
            )
    active_notes = {}
    note_list = []
    for ts in range(pianoroll.shape[1]):
        active = pianoroll[:, ts].nonzero()[0]

        del_notes = []
        for note in active_notes:
            if note not in active:
                del_notes.append(note)

        for note in del_notes:
            note_list.append(active_notes.pop(note))

        for note in active:
            vel = int(pianoroll[note, ts])
            if note not in active_notes:
                active_notes[note] = [note, vel, ts, ts + 1]
            else:
                if vel != active_notes[note][1]:
                    note_list.append(active_notes.pop(note))
                    active_notes[note] = [note, vel, ts, ts + 1]
                else:
                    active_notes[note][-1] += 1

    remaining_active_notes = list(active_notes.keys())
    for note in remaining_active_notes:
        # append any note left
        note_list.append(active_notes.pop(note))

    # Sort array lexicographically by onset, pitch, offset and velocity
    note_list.sort(key=lambda x: (x[2], x[0], x[3], x[1]))

    # Create note array
    note_array = np.array(
        [
            (
                p + init_pitch,
                float(on) / time_div,
                float(off - on) / time_div,
                np.round(vel),
                f"n{i}",
            )
            for i, (p, vel, on, off) in enumerate(note_list)
        ],
        dtype=[
            ("pitch", "i4"),
            (f"onset_{time_unit}", "f4"),
            (f"duration_{time_unit}", "f4"),
            ("velocity", "i4"),
            ("id", "U256"),
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
            raise ValueError(
                "`fields` should be a tuple or a string, but given " f"{type(fields)}"
            )
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
            taix_to_consider = np.array(
                [ti for ti in taix if ti not in matched_target_idxs], dtype=int
            )
            if len(taix_to_consider) > 0:
                # If there are some indices to consider
                candidate_notes = target_note_array[taix_to_consider]

                if check_duration:
                    best_candidate_idx = (
                        candidate_notes[duration_key]
                        - input_note_array[inix][duration_key]
                    ).argmin()
                else:
                    # Take the first one if no other information is given
                    best_candidate_idx = 0

                matched_idxs.append((inix, taix_to_consider[best_candidate_idx]))
                matched_target_idxs.append(taix_to_consider[best_candidate_idx])
        else:
            matched_idxs.append((inix, taix[0]))
            matched_target_idxs.append(taix[0])
    matched_idxs = np.array(matched_idxs)

    warnings.warn(
        "Length of matched idxs: " "{0}".format(len(matched_idxs)), stacklevel=2
    )
    warnings.warn(
        "Length of input note_array: " "{0}".format(len(input_note_array)), stacklevel=2
    )
    warnings.warn(
        "Length of target note_array: " "{0}".format(len(target_note_array)),
        stacklevel=2,
    )

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
    **kwargs,
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
       Indicate from which part do each note come from in the note ids. Default is True.
    **kwargs : dict
         Additional keyword arguments to pass to `utils.music.note_array_from_part()`

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

    is_score = False
    note_array = []
    for i, part in enumerate(part_list):
        if isinstance(part, (Part, PartGroup)):
            # set include_divs_per_quarter, to correctly merge different divs
            kwargs["include_divs_per_quarter"] = True
            is_score = True
            if isinstance(part, Part):
                na = note_array_from_part(
                    part,
                    **kwargs
                )
            elif isinstance(part, PartGroup):
                na = note_array_from_part_list(
                    part.children,
                    unique_id_per_part=unique_id_per_part,
                    **kwargs
                )
        elif isinstance(part, PerformedPart):
            na = part.note_array()
        if unique_id_per_part and len(part_list) > 1:
            # Update id with part number
            na["id"] = np.array(
                ["P{0:02d}_".format(i) + nid for nid in na["id"]], dtype=na["id"].dtype
            )
        note_array.append(na)

    if is_score:
        # rescale if parts have different divs
        divs_per_parts = [part[0]["divs_pq"] for part in note_array]
        lcm = np.lcm.reduce(divs_per_parts)
        time_multiplier_per_part = [int(lcm / d) for d in divs_per_parts]
        for na, time_mult in zip(note_array, time_multiplier_per_part):
            na["onset_div"] = na["onset_div"] * time_mult
            na["duration_div"] = na["duration_div"] * time_mult
            na["divs_pq"] = na["divs_pq"] * time_mult

    # concatenate note_arrays
    note_array = np.hstack(note_array)

    onset_unit, _ = get_time_units_from_note_array(note_array)

    # sort by onset and pitch
    pitch_sort_idx = np.argsort(note_array["pitch"])
    note_array = note_array[pitch_sort_idx]
    onset_sort_idx = np.argsort(note_array[onset_unit], kind="mergesort")
    note_array = note_array[onset_sort_idx]

    return note_array


def rest_array_from_part_list(
    part_list,
    unique_id_per_part=True,
    include_pitch_spelling=False,
    include_key_signature=False,
    include_time_signature=False,
    include_grace_notes=False,
    include_staff=False,
    collapse=False,
):
    """
    Construct a structured Rest array from a list of Part objects

    Parameters
    ----------
    part_list : list
       A list of `Part` or `PerformedPart` objects. All elements in
       the list must be of the same type.
    unique_id_per_part : bool (optional)
       Indicate from which part do each rest come from in the rest ids.
    include_pitch_spelling: bool (optional)
       Include pitch spelling information in rest array.
       This is a dummy attribute and returns zeros everywhere.
       Default is False.
    include_key_signature: bool (optional)
       Include key signature information in output rest array.
       Only valid if parts in `part_list` are `Part` objects.
       See `rest_array_from_part` for more info.
       Default is False.
    include_time_signature : bool (optional)
       Include time signature information in output rest array.
       Only valid if parts in `part_list` are `Part` objects.
       See `rest_array_from_part` for more info.
       Default is False.
    include_grace_notes : bool (optional)
        If `True`,  includes grace note information, i.e. "" for every rest).
        Default is False
    include_staff : bool (optional)
        If `True`,  includes note staff number.
        Default is False

    Returns
    -------
    rest_array: structured array
        A structured array containing pitch (always zero), onset, duration, voice
        and id for each rest in each part of the `part_list`. The rest
        ids in this array include the number of the part to which they
        belong.
    """
    from partitura.score import Part, PartGroup

    rest_array = []
    for i, part in enumerate(part_list):
        if isinstance(part, (Part, PartGroup)):
            if isinstance(part, Part):
                na = rest_array_from_part(
                    part=part,
                    unique_id_per_part=unique_id_per_part,
                    include_pitch_spelling=include_pitch_spelling,
                    include_key_signature=include_key_signature,
                    include_time_signature=include_time_signature,
                    include_grace_notes=include_grace_notes,
                    inlcude_staff=include_staff,
                    collapse=collapse,
                )
            elif isinstance(part, PartGroup):
                na = rest_array_from_part_list(
                    part_list=part.children,
                    unique_id_per_part=unique_id_per_part,
                    include_pitch_spelling=include_pitch_spelling,
                    include_key_signature=include_key_signature,
                    include_time_signature=include_time_signature,
                    include_grace_notes=include_grace_notes,
                    inlcude_staff=include_staff,
                    collapse=collapse,
                )
        if unique_id_per_part:
            # Update id with part number
            na["id"] = np.array(
                ["P{0:02d}_".format(i) + nid for nid in na["id"]], dtype=na["id"].dtype
            )
        rest_array.append(na)

    # concatenate note_arrays
    rest_array = np.hstack(rest_array)

    onset_unit, _ = get_time_units_from_note_array(rest_array)

    # sort by onset and pitch
    pitch_sort_idx = np.argsort(rest_array["pitch"])
    rest_array = rest_array[pitch_sort_idx]
    onset_sort_idx = np.argsort(rest_array[onset_unit], kind="mergesort")
    rest_array = rest_array[onset_sort_idx]

    return rest_array


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
    include_metrical_position=False,
    include_grace_notes=False,
    include_staff=False,
    include_divs_per_quarter=False,
):
    """
    Create a structured array with note information
    from a `Part` object.

    Parameters
    ----------
    part : partitura.score.Part
        An object representing a score part.
    include_pitch_spelling : bool (optional)
        It's a dummy attribute for consistancy between note_array_from_part and note_array_from_part_list.
        Default is False
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
    include_metrical_position : bool (optional)
        If `True`,  includes metrical position information, i.e.,
        the position of the onset time of each note with respect to its
        measure (all notes starting at the same time have the same metrical
        position).
        Default is False
    include_grace_notes : bool (optional)
        If `True`,  includes grace note information, i.e. if a note is a
        grace note and the grace type "" for non grace notes).
        Default is False
    include_staff : bool (optional)
        If `True`,  includes staff information
        Default is False
    include_divs_per_quarter : bool (optional)
        If `True`,  include the number of divs (e.g. MIDI ticks,
        MusicXML ppq) per quarter note of the current part.
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
            * 'ts_mus_beat' : number of musical beats is it's set, otherwise ts_beats

        If `include_metrical_position` is True:
            * 'is_downbeat': 1 if the note onset is on a downbeat, 0 otherwise
            * 'rel_onset_div': number of divs elapsed from the beginning of the note measure
            * 'tot_measure_divs' : total number of divs in the note measure

        If 'include_grace_notes' is True:
            * 'is_grace': 1 if the note is a grace 0 otherwise
            * 'grace_type' : the type of the grace notes "" for non grace notes

        If 'include_staff' is True:
            * 'staff' : the staff number for each note

        If 'include_divs_per_quarter' is True:
            * 'divs_pq': the number of divs per quarter note
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

    if include_metrical_position:
        metrical_position_map = part.metrical_position_map
    else:
        metrical_position_map = None

    if include_divs_per_quarter:
        parts_quarter_times = part._quarter_times
        parts_quarter_durations = part._quarter_durations
        if not len(parts_quarter_durations) == 1:
            raise Exception(
                "Note array from parts with multiple divisions is not supported. Found divisions",
                parts_quarter_durations,
                "at times",
                parts_quarter_times,
            )
        divs_per_quarter = parts_quarter_durations[0]
    else:
        divs_per_quarter = None

    note_array = note_array_from_note_list(
        note_list=part.notes_tied,
        beat_map=part.beat_map,
        quarter_map=part.quarter_map,
        time_signature_map=time_signature_map,
        key_signature_map=key_signature_map,
        metrical_position_map=metrical_position_map,
        include_pitch_spelling=include_pitch_spelling,
        include_grace_notes=include_grace_notes,
        include_staff=include_staff,
        divs_per_quarter=divs_per_quarter,
    )

    return note_array


def rest_array_from_part(
    part,
    include_pitch_spelling=False,
    include_key_signature=False,
    include_time_signature=False,
    include_metrical_position=False,
    include_grace_notes=False,
    include_staff=False,
    collapse=False,
):
    """
    Create a structured array with rest information
    from a `Part` object Similar to note_array.

    Parameters
    ----------
    part : partitura.score.Part
        An object representing a score part.
    include_pitch_spelling : bool (optional)
        It's a dummy attribute for consistancy between rest_array_from_part and rest_array_from_part_list.
        Default is False
    include_pitch_spelling : bool (optional)
        If `True`, includes pitch spelling information for each
        rest.
        This is a dummy attribute returns zeros everywhere.
        Default is False
    include_key_signature : bool (optional)
        If `True`, includes key signature information, i.e.,
        the key signature at the onset time of each rest (all
        notes starting at the same time have the same key signature).
        Default is False
    include_time_signature : bool (optional)
        If `True`,  includes time signature information, i.e.,
        the time signature at the onset time of each rest (all
        rests starting at the same time have the same time signature).
        Default is False
    include_metrical_position : bool (optional)
        If `True`,  includes metrical position information, i.e.,
        the position of the onset time of each note with respect to its
        measure.
        Default is False
    include_grace_notes : bool (optional)
        If `True`,  includes grace note information, i.e. the grace type is "" for all rests).
        Default is False
    collapse : bool (optional)
        If 'True', collapses consecutive rest onsets on the same voice, to a single rest of their combined duration.
        Default is False

    Returns
    -------
    rest_array : structured array
        A structured array containing rest information (pitch is always 0).
    """
    if include_time_signature:
        time_signature_map = part.time_signature_map
    else:
        time_signature_map = None

    if include_key_signature:
        key_signature_map = part.key_signature_map
    else:
        key_signature_map = None

    if include_metrical_position:
        metrical_position_map = part.metrical_position_map
    else:
        metrical_position_map = None

    rest_array = rest_array_from_rest_list(
        rest_list=part.rests,
        beat_map=part.beat_map,
        quarter_map=part.quarter_map,
        time_signature_map=time_signature_map,
        key_signature_map=key_signature_map,
        metrical_position_map=metrical_position_map,
        include_pitch_spelling=include_pitch_spelling,
        include_grace_notes=include_grace_notes,
        include_staff=include_staff,
        collapse=collapse,
    )

    return rest_array


def note_array_from_note_list(
    note_list,
    beat_map=None,
    quarter_map=None,
    time_signature_map=None,
    key_signature_map=None,
    metrical_position_map=None,
    include_pitch_spelling=False,
    include_grace_notes=False,
    include_staff=False,
    divs_per_quarter=None,
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
    metrical_position_map: callable or None (optional)
        A function that maps score time in divs to the position in
        the measure at that time.
        If `None` is given, the output structured array will not
        include the metrical position information.
    include_pitch_spelling : bool (optional)
        If `True`, includes pitch spelling information for each
        note. Default is False
    include_grace_notes : bool (optional)
        If `True`,  includes grace note information, i.e. if a note is a
        grace note has one of the types "appoggiatura, acciaccatura, grace" and
        the grace type "" for non grace notes).
        Default is False
    include_staff : bool (optional)
        If `True`,  includes the staff number for every note.
        Default is False
    divs_per_quarter : int or None (optional)
        The number of divs (e.g. MIDI ticks, MusicXML ppq) per quarter
        note of the current part.
        Default is None


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
            * 'is_grace' : Is the note a grace note. Yes if true.
            * 'grace_type' : The type of grace note. "" for non grace notes.
            * 'ks_fifths': Fifths starting from C in the circle of fifths.
              Included if `key_signature_map` is not `None`.
            * 'mode': major or minor. Included If `key_signature_map` is
              not `None`.
            * 'ts_beats': number of beats in a measure. If `time_signature_map`
               is True.
            * 'ts_beat_type': type of beats (denominator of the time signature).
              If `include_time_signature` is True.
            * 'is_downbeat': 1 if the note onset is on a downbeat, 0 otherwise.
               If `measure_map` is not None.
            * 'rel_onset_div': number of divs elapsed from the beginning of the
               note measure. If `measure_map` is not None.
            * 'tot_measure_div' : total number of divs in the note measure
               If `measure_map` is not None.
            * 'staff' : number of note staff.
            * 'divs_pq' : number of parts per quarter note.
    """

    fields = []
    if beat_map is not None:
        # Preserve the order of the fields
        fields += [("onset_beat", "f4"), ("duration_beat", "f4")]

    if quarter_map is not None:
        fields += [("onset_quarter", "f4"), ("duration_quarter", "f4")]
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

    # fields for pitch spelling
    if include_grace_notes:
        fields += [("is_grace", "b"), ("grace_type", "U256")]

    # fields for key signature
    if key_signature_map is not None:
        fields += [("ks_fifths", "i4"), ("ks_mode", "i4")]

    # fields for time signature
    if time_signature_map is not None:
        fields += [("ts_beats", "i4"), ("ts_beat_type", "i4"), ("ts_mus_beats", "i4")]

    # fields for metrical position
    if metrical_position_map is not None:
        fields += [
            ("is_downbeat", "i4"),
            ("rel_onset_div", "i4"),
            ("tot_measure_div", "i4"),
        ]
    # field for staff
    if include_staff:
        fields += [("staff", "i4")]

    # field for divs_pq
    if divs_per_quarter:
        fields += [("divs_pq", "i4")]

    note_array = []
    for note in note_list:

        note_info = tuple()
        note_on_div = note.start.t
        note_off_div = note.start.t + note.duration_tied
        note_dur_div = note_off_div - note_on_div

        if beat_map is not None:
            note_on_beat, note_off_beat = beat_map([note_on_div, note_off_div])
            note_dur_beat = note_off_beat - note_on_beat

            note_info += (note_on_beat, note_dur_beat)

        if quarter_map is not None:
            note_on_quarter, note_off_quarter = quarter_map([note_on_div, note_off_div])
            note_dur_quarter = note_off_quarter - note_on_quarter

            note_info += (note_on_quarter, note_dur_quarter)

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

        if include_grace_notes:
            is_grace = hasattr(note, "grace_type")
            if is_grace:
                grace_type = note.grace_type
            else:
                grace_type = ""
            note_info += (is_grace, grace_type)

        if key_signature_map is not None:
            fifths, mode = key_signature_map(note.start.t)

            note_info += (fifths, mode)

        if time_signature_map is not None:
            beats, beat_type, mus_beats = time_signature_map(note.start.t)

            note_info += (beats, beat_type, mus_beats)

        if metrical_position_map is not None:
            rel_onset_div, tot_measure_div = metrical_position_map(note.start.t)

            is_downbeat = 1 if rel_onset_div == 0 else 0

            note_info += (is_downbeat, rel_onset_div, tot_measure_div)

        if include_staff:
            note_info += ((note.staff if note.staff else 0),)

        if divs_per_quarter:
            note_info += (divs_per_quarter,)

        note_array.append(note_info)

    note_array = np.array(note_array, dtype=fields)

    # Sanitize voice information
    no_voice_idx = np.where(note_array["voice"] == -1)[0]
    max_voice = note_array["voice"].max()
    note_array["voice"][no_voice_idx] = max_voice + 1

    # sort by onset and pitch
    onset_unit, _ = get_time_units_from_note_array(note_array)
    pitch_sort_idx = np.argsort(note_array["pitch"])
    note_array = note_array[pitch_sort_idx]
    onset_sort_idx = np.argsort(note_array[onset_unit], kind="mergesort")
    note_array = note_array[onset_sort_idx]

    return note_array


def rest_array_from_rest_list(
    rest_list,
    beat_map=None,
    quarter_map=None,
    time_signature_map=None,
    key_signature_map=None,
    metrical_position_map=None,
    include_pitch_spelling=False,
    include_grace_notes=False,
    include_staff=False,
    collapse=False,
):
    """
    Create a structured array with rest information
    from a list of `Rest` objects.

    Parameters
    ----------
    rest_list : list of `Rest` objects
        A list of `Rest` objects containing score information.
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
    metrical_position_map: callable or None (optional)
        A function that maps score time in divs to the position in
        the measure at that time.
        If `None` is given, the output structured array will not
        include the metrical position information.
    include_pitch_spelling : bool (optional)
        If `True`, includes pitch spelling information for each
        rest. This is a dummy attribute and returns zeros everywhere.
        Default is False
    include_grace_notes : bool (optional)
        If `True`,  includes grace note information, i.e. "" for all rests).
        Default is False
    include_staff : bool (optional)
        If `True`,  includes the staff number for every note.
        Default is False
    collapse : bool (optional)
        If `True`, joins rests on consecutive onsets on the same voice and combines their durations.
        Keeps the id of the first one.
        Default is False

    Returns
    -------
    rest_array : structured array
        A structured array containing rest information. Pitch is set to 0.
    """

    fields = []
    if beat_map is not None:
        # Preserve the order of the fields
        fields += [("onset_beat", "f4"), ("duration_beat", "f4")]

    if quarter_map is not None:
        fields += [("onset_quarter", "f4"), ("duration_quarter", "f4")]
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

    # fields for pitch spelling
    if include_grace_notes:
        fields += [("is_grace", "b"), ("grace_type", "U256")]

    # fields for key signature
    if key_signature_map is not None:
        fields += [("ks_fifths", "i4"), ("ks_mode", "i4")]

    # fields for time signature
    if time_signature_map is not None:
        fields += [("ts_beats", "i4"), ("ts_beat_type", "i4")]

    # fields for metrical position
    if metrical_position_map is not None:
        fields += [
            ("is_downbeat", "i4"),
            ("rel_onset_div", "i4"),
            ("tot_measure_div", "i4"),
        ]
    # fields for staff
    if include_staff:
        fields += [("staff", "i4")]

    rest_array = []
    for rest in rest_list:

        rest_info = tuple()
        rest_on_div = rest.start.t
        rest_off_div = rest.start.t + rest.duration_tied
        rest_dur_div = rest_off_div - rest_on_div

        if beat_map is not None:
            note_on_beat, note_off_beat = beat_map([rest_on_div, rest_off_div])
            note_dur_beat = note_off_beat - note_on_beat

            rest_info += (note_on_beat, note_dur_beat)

        if quarter_map is not None:
            note_on_quarter, note_off_quarter = quarter_map([rest_on_div, rest_off_div])
            note_dur_quarter = note_off_quarter - note_on_quarter

            rest_info += (note_on_quarter, note_dur_quarter)

        rest_info += (
            rest_on_div,
            rest_dur_div,
            0,
            rest.voice if rest.voice is not None else -1,
            rest.id,
        )

        if include_pitch_spelling:
            step = 0
            alter = 0
            octave = 0

            rest_info += (step, alter, octave)

        if include_grace_notes:
            is_grace = hasattr(rest, "grace_type")
            if is_grace:
                grace_type = rest.grace_type
            else:
                grace_type = ""
            rest_info += (is_grace, grace_type)

        if key_signature_map is not None:
            fifths, mode = key_signature_map(rest.start.t)

            rest_info += (fifths, mode)

        if time_signature_map is not None:
            beats, beat_type = time_signature_map(rest.start.t)

            rest_info += (beats, beat_type)

        if metrical_position_map is not None:
            rel_onset_div, tot_measure_div = metrical_position_map(rest.start.t)

            is_downbeat = 1 if rel_onset_div == 0 else 0

            rest_info += (is_downbeat, rel_onset_div, tot_measure_div)

        if include_staff:
            rest_info += ((rest.staff if rest.staff else 0),)

        rest_array.append(rest_info)

    rest_array = np.array(rest_array, dtype=fields)

    # Sanitize voice information
    if rest_list:
        no_voice_idx = np.where(rest_array["voice"] == -1)[0]
        max_voice = rest_array["voice"].max()
        rest_array["voice"][no_voice_idx] = max_voice + 1

    # sort by onset and pitch
    onset_unit, _ = get_time_units_from_note_array(rest_array)
    pitch_sort_idx = np.argsort(rest_array["pitch"])
    rest_array = rest_array[pitch_sort_idx]
    onset_sort_idx = np.argsort(rest_array[onset_unit], kind="mergesort")
    rest_array = rest_array[onset_sort_idx]

    if collapse:
        rest_array = rec_collapse_rests(rest_array)
    return rest_array


def collapse_rests(rest_array):
    filter_idx = []
    output_idx = []
    for i, rest in enumerate(rest_array):
        if i not in filter_idx:
            idxs = np.where(
                (rest_array["onset_beat"] == rest["onset_beat"] + rest["duration_beat"])
                & (rest_array["voice"] == rest["voice"])
            )[0]
            for idx in idxs:
                rest_array[i]["duration_beat"] = (
                    rest["duration_beat"] + rest_array[idx]["duration_beat"]
                )
                rest_array[i]["duration_div"] = (
                    rest["duration_div"] + rest_array[idx]["duration_div"]
                )
                filter_idx.append(idx)
            output_idx.append(i)
    return rest_array[output_idx], filter_idx


def rec_collapse_rests(rest_array):
    cond = True
    while cond:
        rest_array, filter_idx = collapse_rests(rest_array)
        cond = len(filter_idx) > 0
    return rest_array


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


def performance_from_part(part, bpm=100, velocity=64):
    """
    Create a PerformedPart object from a Part object

    Parameters
    ----------
    part: Part
        The part from which we want to generate a performed part
    bpm : float
        Beats per minute
    velocity: float or int
        The MIDI velocity for all notes.

    Returns
    -------
    ppart: PerformedPart

    Potential extensions
    --------------------
    * allow for bpm to be a callable or an 2D array with columns (onset, bpm)
    * allow for velocity to be a callable or a 2D array (onset, velocity)
    """
    from partitura.score import Part
    from partitura.performance import PerformedPart

    if not isinstance(part, Part):
        raise ValueError(
            "The input `part` must be a "
            f"`partitura.score.Part` instance, not {type(part)}"
        )

    ppart_fields = [
        ("onset_sec", "f4"),
        ("duration_sec", "f4"),
        ("pitch", "i4"),
        ("velocity", "i4"),
        ("track", "i4"),
        ("channel", "i4"),
        ("id", "U256"),
    ]
    snote_array = part.note_array()

    pnote_array = np.zeros(len(snote_array), dtype=ppart_fields)

    unique_onsets = np.unique(snote_array["onset_beat"])
    # Cast as object to avoid warnings, but seems to work well
    # in numpy version 1.20.1
    unique_onset_idxs = np.array(
        [np.where(snote_array["onset_beat"] == u)[0] for u in unique_onsets],
        dtype=object,
    )

    iois = np.diff(unique_onsets)

    bp = 60 / float(bpm)

    # TODO: allow for variable bpm and velocity
    pnote_array["duration_sec"] = bp * snote_array["duration_beat"]
    pnote_array["velocity"] = int(velocity)
    pnote_array["pitch"] = snote_array["pitch"]
    pnote_array["id"] = snote_array["id"]
    p_onsets = np.r_[0, np.cumsum(iois * bp)]

    for ix, on in zip(unique_onset_idxs, p_onsets):
        # ix has to be cast as integer depending on the
        # numpy version...
        pnote_array["onset_sec"][ix.astype(int)] = on

    ppart = PerformedPart.from_note_array(pnote_array)

    return ppart


def get_time_maps_from_alignment(
    ppart_or_note_array, spart_or_note_array, alignment, remove_ornaments=True
):
    """
    Get time maps to convert performance time (in seconds) to score time (in beats)
    and visceversa.

    Parameters
    ----------
    ppart_or_note_array : PerformedPart or structured array
        The performance information as either PerformedPart or the
        note_array generated from such an object.
    spart_or_note_array : Part or structured array
        Score information as either a Part object or the note array
        generated from such an object.
    alignment : list
        The score--performance alignment, a list of dictionaries.
        (see `partitura.io.importmatch.alignment_from_matchfile` for reference)
    remove_ornaments : bool (optional)
        Whether to consider or not ornaments (including grace notes)

    Returns
    -------
    ptime_to_stime_map : scipy.interpolate.interp1d
        An instance of interp1d (a callable) that maps performance time (in seconds)
        to score time (in beats).
    stime_to_ptime_map : scipy.interpolate.interp1d
        An instance of inter1d (a callable) that maps score time (in beats) to
        performance time (in seconds).

    Note
    ----
    This methods uses the average value of the score onsets of notes that are
    written in the score as part of a chord (i.e., which start at the same time).
    """
    # Ensure that we are using structured note arrays
    perf_note_array = ensure_notearray(ppart_or_note_array)
    score_note_array = ensure_notearray(spart_or_note_array)

    # Get indices of the matched notes (notes in the score
    # for which there is a performance note
    match_idx = get_matched_notes(score_note_array, perf_note_array, alignment)

    # Get onsets and durations
    score_onsets = score_note_array[match_idx[:, 0]]["onset_beat"]
    score_durations = score_note_array[match_idx[:, 0]]["duration_beat"]

    perf_onsets = perf_note_array[match_idx[:, 1]]["onset_sec"]

    # Use only unique onsets
    score_unique_onsets = np.unique(score_onsets)

    # Remove grace notes
    if remove_ornaments:
        # TODO: check that all onsets have a duration?
        # ornaments (grace notes) do not have a duration
        score_unique_onset_idxs = np.array(
            [
                np.where(np.logical_and(score_onsets == u, score_durations > 0))[0]
                for u in score_unique_onsets
            ],
            dtype=object,
        )

    else:
        score_unique_onset_idxs = np.array(
            [np.where(score_onsets == u)[0] for u in score_unique_onsets],
            dtype=object,
        )

    # For chords, we use the average performed onset as a proxy for
    # representing the "performeance time" of the position of the score
    # onsets
    eq_perf_onsets = np.array(
        [np.mean(perf_onsets[u]) for u in score_unique_onset_idxs]
    )

    # Get maps
    ptime_to_stime_map = interp1d(
        x=eq_perf_onsets,
        y=score_unique_onsets,
        bounds_error=False,
        fill_value="extrapolate",
    )
    stime_to_ptime_map = interp1d(
        y=eq_perf_onsets,
        x=score_unique_onsets,
        bounds_error=False,
        fill_value="extrapolate",
    )

    return ptime_to_stime_map, stime_to_ptime_map


def get_matched_notes(spart_note_array, ppart_note_array, alignment):
    """
    Get the indices of the matched notes in an alignment

    Parameters
    ----------
    spart_note_array : structured numpy array
        note_array of the score part
    ppart_note_array : structured numpy array
        note_array of the performed part
    alignment : list
        The score--performance alignment, a list of dictionaries.
        (see `partitura.io.importmatch.alignment_from_matchfile` for reference)

    Returns
    -------
    matched_idxs : np.ndarray
        A 2D array containing the indices of the matched score and
        performed notes, where the columns are
        (index_in_score_note_array, index_in_performance_notearray)
    """
    # Get matched notes
    matched_idxs = []
    for al in alignment:
        # Get only matched notes (i.e., ignore inserted or deleted notes)
        if al["label"] == "match":

            # if ppart_note_array['id'].dtype != type(al['performance_id']):
            if not isinstance(ppart_note_array["id"], type(al["performance_id"])):
                p_id = str(al["performance_id"])
            else:
                p_id = al["performance_id"]

            p_idx = int(np.where(ppart_note_array["id"] == p_id)[0])

            s_idx = np.where(spart_note_array["id"] == al["score_id"])[0]

            if len(s_idx) > 0:
                s_idx = int(s_idx)
                matched_idxs.append((s_idx, p_idx))

    return np.array(matched_idxs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
