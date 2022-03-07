import sys
import warnings
import numpy as np
from scipy.interpolate import interp1d
import types
from typing import List, Union, Tuple

import partitura.score as score
from partitura.utils import ensure_notearray


class InvalidBasisException(Exception):
    pass


def print_basis_functions():
    """Print a list of all basisfunction names defined in this module,
    with descriptions where available.

    """
    module = sys.modules[__name__]
    doc_indent = 4
    for name in list_basis_functions():
        print('* {}'.format(name))
        member = getattr(sys.modules[__name__], name)
        if member.__doc__:
            print(' ' * doc_indent + member.__doc__.replace('\n', ' ' * doc_indent + '\n'))


def list_basis_functions():
    """Return a list of all basisfunction names defined in this module.

    The basisfunction names listed here can be specified by name in
    the `make_basis` function. For example:

    >>> basis, names = make_note_feats(part, ['metrical_basis', 'articulation_basis'])

    Returns
    -------
    list
        A list of strings

    """
    module = sys.modules[__name__]
    bfs = []
    exclude = {'make_basis'}
    for name in dir(module):
        if name in exclude:
            continue
        member = getattr(sys.modules[__name__], name)
        if isinstance(member, types.FunctionType) and name.endswith('_basis'):
            bfs.append(name)
    return bfs


def make_note_feats(part: Union[score.Part, score.PartGroup, List], basis_functions: Union[List, str]) -> Tuple[np.ndarray, List]:
    """Compute the specified basis functions for a part.

    The function returns the computed basis functions as a N x M
    array, where N equals `len(part.notes_tied)` and M equals the
    total number of descriptors of all basis functions that occur in
    part.

    Furthermore the function returns the names of the basis functions.
    A list of strings of size M. The names have the name of the
    function prepended to the name of the descriptor. For example if a
    function named `abc_basis` returns descriptors `a`, `b`, and `c`,
    then the list of names returned by `make_basis(part,
    ['abc_basis'])` will be ['abc_basis.a', 'abc_basis.b',
    'abc_basis.c'].

    Parameters
    ----------
    part : Part
        The score as a Part instance
    basis_functions : list or str
        A list of basis functions. Elements of the list can be either
        the functions themselves or the names of a basis function as
        strings (or a mix). The basis functions specified by name are
        looked up in the `basismixer.basisfunctions` module.

    Returns
    -------
    basis : ndarray
        The basis functions
    names : list
        The basis names

    """
    part = score.merge_parts(part)
    na = ensure_notearray(part, include_metrical_position=True , include_grace_notes=True, include_time_signature=True)
    acc = []
    if isinstance(basis_functions, str) and basis_functions=="all":
        basis_functions = list_basis_functions()
    elif not isinstance(basis_functions, list):
        raise TypeError("basis_functions variable {} needs to be list or all".format(basis_functions))

    for bf in basis_functions:
        if isinstance(bf, str):
            # get function by name from module
            func = getattr(sys.modules[__name__], bf)
        elif isinstance(bf, types.FunctionType):
            func = bf
        else:
            warnings.warn('Ignoring unknown basis function {}'.format(bf))
        bf, bn = func(na, part)

        # check if the size and number of the basis function are correct
        if bf.size != 0 :
            if bf.shape[1] != len(bn):
                msg = ('number of basis names {} does not equal '
                       'number of basis {}'.format(len(bn), bf.shape[1]))
                raise InvalidBasisException(msg)
            n_notes = len(part.notes_tied)
            if len(bf) != n_notes:
                msg = ('length of basis {} does not equal '
                       'number of notes {}'.format(len(bf), n_notes))
                raise InvalidBasisException(msg)

            if np.any(np.logical_or(np.isnan(bf), np.isinf(bf))):
                problematic = np.unique(np.where(np.logical_or(np.isnan(bf), np.isinf(bf)))[1])
                msg = ('NaNs or Infs found in the following basis: {} '
                       .format(', '.join(np.array(bn)[problematic])))
                raise InvalidBasisException(msg)

            # prefix basis names by function name
            bn = ['{}.{}'.format(func.__name__, n) for n in bn]

            acc.append((bf, bn))

    _data, _names = zip(*acc)
    basis_data = np.column_stack(_data)
    basis_names = [n for ns in _names for n in ns]
    return basis_data, basis_names


def polynomial_pitch_basis(na, part):
    """Normalize pitch feature.

    """
    pitches = na["pitch"].astype(np.float)
    basis_names = ['pitch']
    max_pitch = 127
    W = pitches / max_pitch
    return np.expand_dims(W, axis=1), basis_names


def duration_basis(na, part):
    """Duration basis.

    Parameters
    ----------
    na : structured array
        The Note array for Unified part.
    """

    basis_names = ['duration']


    durations_beat = na["duration_beat"]
    W = durations_beat
    W.shape = (-1, 1)
    return W, basis_names


def onset_basis(na, part):
    """Onset basis

    Returns:
    * onset : the onset of the note in beats
    * score_position : position of the note in the score between 0 (the beginning of the piece) and 1 (the end of the piece)

    TODO:
    * rel_position_repetition
    """
    basis_names = ['onset', 'score_position']

    onsets_beat = na["onset_beat"]
    rel_position = normalize(onsets_beat, method='minmax')

    W = np.column_stack((onsets_beat, rel_position))

    return W, basis_names


def relative_score_position_basis(na, part):
    W, names = onset_basis(na, part)
    return W[:, 1:], names[1:]


def grace_basis(na, part):
    """Grace basis.

    Returns:
    * grace_note : 1 when the note is a grace note, 0 otherwise
    * n_grace : the length of the grace note sequence to which
                this note belongs (0 for non-grace notes)
    * grace_pos : the (1-based) position of the grace note in
                  the sequence (0 for non-grace notes)

    """

    basis_names = ['grace_note', 'n_grace', 'grace_pos']


    W = np.zeros((len(na), 3))
    W[:, 0] = na["is_grace"]
    grace_notes = na[np.nonzero(na["is_grace"])]
    notes = {n.id:n for n in part.notes_tied}
    indices = np.nonzero(na["is_grace"])[0]
    for i, index in enumerate(indices):
        grace = grace_notes[i]
        n_grace = np.count_nonzero(grace_notes["onset_beat"] == grace["onset_beat"])
        W[index, 1] = n_grace
        W[index, 2] = n_grace - sum(1 for _ in notes[grace["id"]].iter_grace_seq()) + 1
    return W, basis_names


def loudness_direction_basis(na, part):
    """The loudness directions in part.

    This function returns a varying number of descriptors, depending
    on which directions are present. Some directions are grouped
    together. For example 'decrescendo' and 'diminuendo' are encoded
    together in a descriptor 'loudness_decr'. The descriptor names of
    textual directions such as 'adagio' are the verbatim directions.

    Some possible descriptors:
    * p : piano
    * f : forte
    * pp : pianissimo
    * loudness_incr : crescendo direction
    * loudness_decr : decrescendo or diminuendo direction

    """

    onsets = na["onset_div"]
    N = len(onsets)

    directions = list(part.iter_all(
        score.LoudnessDirection, include_subclasses=True))

    def to_name(d):
        if isinstance(d, score.ConstantLoudnessDirection):
            return d.text
        elif isinstance(d, score.ImpulsiveLoudnessDirection):
            return d.text
        elif isinstance(d, score.IncreasingLoudnessDirection):
            return 'loudness_incr'
        elif isinstance(d, score.DecreasingLoudnessDirection):
            return 'loudness_decr'

    basis_by_name = {}
    for d in directions:
        j, bf = basis_by_name.setdefault(to_name(d),
                                         (len(basis_by_name), np.zeros(N)))
        bf += basis_function_activation(d)(onsets)

    W = np.empty((len(onsets), len(basis_by_name)))
    names = [None] * len(basis_by_name)
    for name, (j, bf) in basis_by_name.items():
        W[:, j] = bf
        names[j] = name

    return W, names


def tempo_direction_basis(na, part):
    """The tempo directions in part.

    This function returns a varying number of descriptors, depending
    on which directions are present. Some directions are grouped
    together. For example 'adagio' and 'molto adagio' are encoded
    together in a descriptor 'adagio'.

    Some possible descriptors:
    * adagio : directions like 'adagio', 'molto adagio'

    """
    onsets = na["onset_div"]
    N = len(onsets)

    directions = list(part.iter_all(
        score.TempoDirection, include_subclasses=True))

    def to_name(d):
        if isinstance(d, score.ResetTempoDirection):
            ref = d.reference_tempo
            if ref:
                return ref.text
            else:
                return d.text
        elif isinstance(d, score.ConstantTempoDirection):
            return d.text
        elif isinstance(d, score.IncreasingTempoDirection):
            return 'tempo_incr'
        elif isinstance(d, score.DecreasingTempoDirection):
            return 'tempo_decr'

    basis_by_name = {}
    for d in directions:
        j, bf = basis_by_name.setdefault(to_name(d),
                                         (len(basis_by_name), np.zeros(N)))
        bf += basis_function_activation(d)(onsets)

    W = np.empty((len(onsets), len(basis_by_name)))
    names = [None] * len(basis_by_name)
    for name, (j, bf) in basis_by_name.items():
        W[:, j] = bf
        names[j] = name

    return W, names


def articulation_direction_basis(na, part):
    """
    """
    onsets = na["onset_div"]
    N = len(onsets)

    directions = list(part.iter_all(
        score.ArticulationDirection, include_subclasses=True))

    def to_name(d):
        return d.text

    basis_by_name = {}

    for d in directions:
        j, bf = basis_by_name.setdefault(to_name(d),
                                         (len(basis_by_name), np.zeros(N)))
        bf += basis_function_activation(d)(onsets)

    W = np.empty((len(onsets), len(basis_by_name)))
    names = [None] * len(basis_by_name)

    for name, (j, bf) in basis_by_name.items():
        W[:, j] = bf
        names[j] = name

    return W, names


def basis_function_activation(direction):
    epsilon = 1e-6

    if isinstance(direction, (score.DynamicLoudnessDirection,
                              score.DynamicTempoDirection)):
        # a dynamic direction will be encoded as a ramp from d.start.t to
        # d.end.t, and then a step from d.end.t to the start of the next
        # constant direction.

        # There are two potential issues:

        # Issue 1. d.end is None (e.g. just a ritardando without dashes). In this case
        if direction.end:
            direction_end = direction.end.t
        else:
            # assume the end of d is the end of the measure:
            measure = next(direction.start.iter_prev(score.Measure, eq=True), None)
            if measure:
                direction_end = measure.start.t
            else:
                # no measure, unlikely, but not impossible.
                direction_end = direction.start.t

        if isinstance(direction, score.TempoDirection):
            next_dir = next(direction.start.iter_next(
                score.ConstantTempoDirection), None)
        if isinstance(direction, score.ArticulationDirection):
            next_dir = next(direction.start.iter_next(
                score.ConstantArticulationDirection), None)
        else:
            next_dir = next(direction.start.iter_next(
                score.ConstantLoudnessDirection), None)

        if next_dir:
            # TODO: what do we do when next_dir is too far away?
            sustained_end = next_dir.start.t
        else:
            # Issue 2. there is no next constant direction. In that case the
            # basis function will be a ramp with a quarter note ramp
            sustained_end = direction_end + direction.start.quarter

        x = [direction.start.t,
             direction_end - epsilon,
             sustained_end - epsilon]
        y = [0, 1, 1]

    elif isinstance(direction, (score.ConstantLoudnessDirection,
                                score.ConstantArticulationDirection,
                                score.ConstantTempoDirection)):
        x = [direction.start.t - epsilon,
             direction.start.t,
             direction.end.t - epsilon,
             direction.end.t]
        y = [0, 1, 1, 0]

    else:  # impulsive
        x = [direction.start.t - epsilon,
             direction.start.t,
             direction.start.t + epsilon]
        y = [0, 1, 0]

    return interp1d(x, y, bounds_error=False, fill_value=0)


def slur_basis(na, part):
    """Slur basis.

    Returns:
    * slur_incr : a ramp function that increases from 0
                  to 1 over the course of the slur
    * slur_decr : a ramp function that decreases from 1
                  to 0 over the course of the slur

    """
    names = ['slur_incr', 'slur_decr']
    onsets = na["onset_div"]
    slurs = part.iter_all(score.Slur)
    W = np.zeros((len(onsets), 2))

    for slur in slurs:
        if not slur.end:
            continue
        x = [slur.start.t, slur.end.t]
        y_inc = [0, 1]
        y_dec = [1, 0]
        W[:, 0] += interp1d(x, y_inc, bounds_error=False, fill_value=0)(onsets)
        W[:, 1] += interp1d(x, y_dec, bounds_error=False, fill_value=0)(onsets)

    return W, names


def articulation_basis(na, part):
    """Articulation basis.

    This basis returns articulation-related note annotations, such as accents, legato, and tenuto.

    Possible descriptors:
    * accent : 1 when the note has an annotated accent sign
    * legato : 1 when the note has an annotated legato sign
    * staccato : 1 when the note has an annotated staccato sign
    ...

    """
    names = ['accent', 'strong-accent', 'staccato', 'tenuto',
             'detached-legato', 'staccatissimo', 'spiccato',
             'scoop', 'plop', 'doit', 'falloff', 'breath-mark',
             'caesura', 'stress', 'unstress', 'soft-accent']
    basis_by_name = {}
    notes = part.notes_tied
    N = len(notes)
    for i, n in enumerate(notes):
        if n.articulations:
            for art in n.articulations:
                if art in names:
                    j, bf = basis_by_name.setdefault(
                        art,
                        (len(basis_by_name), np.zeros(N)))
                    bf[i] = 1

    M = len(basis_by_name)
    W = np.empty((N, M))
    names = [None] * M

    for name, (j, bf) in basis_by_name.items():
        W[:, j] = bf
        names[j] = name

    return W, names


# # for a subset of the articulations do e.g.
# def staccato_basis(part):
#     W, names = articulation_basis(part)
#     if 'staccato' in names:
#         i = names.index('staccato')
#         return W[:, i:i + 1], ['staccato']
#     else:
#         return np.empty(len(W)), []


def fermata_basis(na, part):
    """Fermata basis.

    Returns:
    * fermata : 1 when the note coincides with a fermata sign.

    """
    names = ['fermata']
    onsets = na["onset_div"]
    W = np.zeros((len(onsets), 1))
    for ferm in part.iter_all(score.Fermata):
        W[onsets == ferm.start.t, 0] = 1
    return W, names


def metrical_basis(na, part):
    """Metrical basis

    This basis encodes the metrical position in the bar. For example
    the first beat in a 3/4 meter is encoded in a binary descriptor
    'metrical_3_4_0', the fifth beat in a 6/8 meter as
    'metrical_6_8_4', etc. Any positions that do not fall on a beat
    are encoded in a basis suffixed '_weak'. For example a note
    starting on the second 8th note in a bar of 4/4 meter will have a
    non-zero value in the 'metrical_4_4_weak' descriptor.

    """
    notes = part.notes_tied
    ts_map = part.time_signature_map
    bm = part.beat_map
    basis_by_name = {}
    eps = 10 ** -6

    for i, n in enumerate(notes):

        beats, beat_type = ts_map(n.start.t).astype(int)
        measure = next(n.start.iter_prev(score.Measure, eq=True), None)

        if measure:
            measure_start = measure.start.t
        else:
            measure_start = 0

        pos = bm(n.start.t) - bm(measure_start)

        if pos % 1 < eps:
            name = 'metrical_{}_{}_{}'.format(beats, beat_type, int(pos))
        else:
            name = 'metrical_{}_{}_weak'.format(beats, beat_type)

        j, bf = basis_by_name.setdefault(name,
                                         (len(basis_by_name), np.zeros(len(notes))))
        bf[i] = 1

    W = np.empty((len(notes), len(basis_by_name)))
    names = [None] * len(basis_by_name)
    for name, (j, bf) in basis_by_name.items():
        W[:, j] = bf
        names[j] = name

    return W, names


def metrical_strength_basis(na, part):
    """Metrical strength basis

    This basis encodes the beat phase (relative position of a note within
    the measure), as well as metrical strength of common time signatures.
    """
    names = ['beat_phase',
             'metrical_strength_downbeat',
             'metrical_strength_secondary',
             'metrical_strength_weak']

    relod = na["rel_onset_div"].astype(float)
    totmd = na["tot_measure_div"].astype(float)
    W = np.zeros((len(na), len(names)))
    W[:, 0] = np.divide(relod, totmd)  # Onset Phase
    W[:, 1] = na["is_downbeat"].astype(float)
    W[:, 2][W[:, 0] == 0.5] = 1.00
    W[:, 3][np.nonzero(np.add(W[:, 1], W[:, 0]) == 1.00)]

    # TODO re-evaluate decision of using sec_beat and rest
    # notes = part.notes_tied
    # ts_map = part.time_signature_map
    # bm = part.beat_map
    # names = ['beat_phase',
    #          'metrical_strength_downbeat',
    #          'metrical_strength_secondary',
    #          'metrical_strength_weak']
    # W = np.zeros((len(notes), len(names)))
    # W[:, 1] = na["is_downbeat"].as_type(float)
    # for i, n in enumerate(notes):
    #     beats, beat_type = ts_map(n.start.t).astype(int)
    #     measure = next(n.start.iter_prev(score.Measure, eq=True), None)
    #     if beats == 4: # for 4/4
    #         sec_beat = 2
    #     elif beats == 6: # for 6/8
    #         sec_beat = 3
    #     elif beats == 12: # for 12/8
    #         sec_beat = 6
    #     else:
    #         sec_beat = None
    #     measure_start = measure.start.t if measure else 0
    #     pos = bm(n.start.t) - bm(measure_start)
    #     m_pos = np.mod(pos, beats)
    #     W[i, 0] = m_pos / beats
    #     if m_pos == 0:
    #         W[i, 1] = 1
    #     elif m_pos == sec_beat:
    #         W[i, 2] = 1
    #     else:
    #         W[i, 3] = 1

    return W, names


def time_signature_basis(na, part):
    """TIme Signature basis
    This basis encodes the time signature of the note in two sets of one-hot vectors,
    a one hot encoding of number of beats and a one hot encoding of beat type
    """

    notes = ensure_notearray(part)
    ts_map = part.time_signature_map
    possible_beats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'other']
    possible_beat_types = [1, 2, 4, 8, 16, 'other']
    W_beats = np.zeros((len(notes), len(possible_beats)))
    W_types = np.zeros((len(notes), len(possible_beat_types)))

    names = (['time_signature_num_{0}'.format(b) for b in possible_beats] +
             ['time_signature_den_{0}'.format(b) for b in possible_beat_types])

    for i, n in enumerate(notes):
        beats, beat_type = ts_map(n["onset_div"]).astype(int)

        if beats in possible_beats:
            W_beats[i, beats - 1] = 1
        else:
            W_beats[i, -1] = 1

        if beat_type in possible_beat_types:
            W_types[i, possible_beat_types.index(beat_type)] = 1
        else:
            W_types[i, -1] = 1

    W = np.column_stack((W_beats, W_types))

    return W, names


def vertical_neighbor_basis(na, part):
    """Vertical neighbor basis.

    Describes various aspects of simultaneously starting notes.

    Returns:
    * n_total :
    * n_above :
    * n_below :
    * highest_pitch :
    * lowest_pitch :
    * pitch_range :

    """
    # the list of descriptors
    names = ['n_total', 'n_above', 'n_below',
             'highest_pitch', 'lowest_pitch', 'pitch_range']
    W = np.empty((len(na), len(names)))
    for i, n in enumerate(na):
        neighbors = na[np.where(na["onset_beat"] == n["onset_beat"])]["pitch"]
        max_pitch = np.max(neighbors)
        min_pitch = np.min(neighbors)
        W[i, 0] = len(neighbors) - 1
        W[i, 1] = np.sum(neighbors > n["pitch"])
        W[i, 2] = np.sum(neighbors < n["pitch"])
        W[i, 3] = max_pitch
        W[i, 4] = min_pitch
        W[i, 5] = max_pitch - min_pitch
    return W, names


def normalize(data, method='minmax'):
    """
    Normalize data in one of several ways.

    The available normalization methods are:

    * minmax
      Rescale `data` to the range `[0, 1]` by subtracting the minimum
      and dividing by the range. If `data` is a 2d array, each column is
      rescaled to `[0, 1]`.

    * tanh
      Rescale `data` to the interval `(-1, 1)` using `tanh`. Note that
      if `data` is non-negative, the output interval will be `[0, 1)`.

    * tanh_unity
      Like "soft", but rather than rescaling strictly to the range (-1,
      1), following will hold:

      normalized = normalize(data, method="tanh_unity")
      np.where(data==1) == np.where(normalized==1)

      That is, the normalized data will equal one wherever the original data
      equals one. The target interval is `(-1/np.tanh(1), 1/np.tanh(1))`.

    Parameters
    ----------
    data: ndarray
        Data to be normalized
    method: {'minmax', 'tanh', 'tanh_unity'}, optional
        The normalization method. Defaults to 'minmax'.

    Returns
    -------
    ndarray
        Normalized copy of the data
    """

    """Normalize the data in `data`. There are several normalization

    """
    if method == 'minmax':
        vmin = np.min(data, 0)
        vmax = np.max(data, 0)

        if np.isclose(vmin, vmax):
            # Return all values as 0 or as 1?
            return np.zeros_like(data)
        else:
            return (data - vmin) / (vmax - vmin)
    elif method == 'tanh':
        return np.tanh(data)
    elif method == 'tanh_unity':
        return np.tanh(data) / np.tanh(1)
