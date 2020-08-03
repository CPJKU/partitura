#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spiral array representation and tonal tension profiles using Herreman and Chew's tension ribbons

References
----------

.. [1] D. Herremans and E. Chew (2016) Tension ribbons: Quantifying and visualising tonal tension.
       Proceedings of the Second International Conference on Technologies for Music Notation and
       Representation (TENOR), Cambridge, UK.
"""
import numpy as np
import scipy.spatial.distance as distance

from partitura.score import Part, PartGroup
from partitura.utils.music import key_int_to_mode

# Scaling factors
A = np.sqrt(2. / 15.) * np.pi / 2.0
R = 1.0

# From Elaine Chew's thesis
DEFAULT_WEIGHTS = np.array([0.516, 0.315, 0.168])
ALPHA = 0.75
BETA = 0.75

STEPS_BY_FIFTHS = ['F', 'C', 'G', 'D', 'A', 'E', 'B']

NOTES_BY_FIFTHS = []
for alt in range(-4, 5):
    NOTES_BY_FIFTHS += [(step, alt) for step in STEPS_BY_FIFTHS]

# Index of C
C_IDX = NOTES_BY_FIFTHS.index(('C', 0))
# C lies in the middle of the spiral array
T = (np.arange(len(NOTES_BY_FIFTHS)) - C_IDX) * np.pi / 2.


def e_distance(x, y):
    """
    Euclidean distance between two points
    """
    return np.sqrt(((x - y) ** 2).sum())


def helical_to_cartesian(t, r=R, a=A):
    """
    Transform helical coordinates to cartesian
    """
    x = r * np.sin(t)
    y = r * np.cos(t)
    z = a * t

    return x, y, z

def ensure_norm(x):
    """
    Ensure that vectors are normalized
    """
    if not np.isclose(x.sum(), 1):
        return x / x.sum()
    else:
        return x


X, Y, Z = helical_to_cartesian(T)
PITCH_COORDINATES = np.column_stack((X, Y, Z))
MAJOR_IDXS = np.array([0, 1, 4], dtype=int)
MINOR_IDXS = np.array([0, 1, -3], dtype=int)

# The scaling factor is the distance between C and B#, as used
# in Cancino-Chacón and Grachten (2018)
SCALE_FACTOR = 1.0 / e_distance(PITCH_COORDINATES[C_IDX],
                                PITCH_COORDINATES[NOTES_BY_FIFTHS.index(('B', 1))])


def major_chord(tonic_idx, w=DEFAULT_WEIGHTS):
    """
    Major chord representation in the spiral array space.

    Parameters
    ----------
    tonic_idx : int
        Index of the root of the chord in NOTES_BY_FIFTHS
    w : array
        3D vector containing the tonal weights. Default is DEFAULT_WEIGHTS.

    Returns
    -------
    chord : array
        Vector representation of the chord
    """
    return np.dot(ensure_norm(w), PITCH_COORDINATES[MAJOR_IDXS + tonic_idx])


def minor_chord(tonic_idx, w=DEFAULT_WEIGHTS):
    """
    Minor chord representation in the spiral array space.

    Parameters
    ----------
    tonic_idx : int
        Index of the root of the chord in NOTES_BY_FIFTHS
    w : array
        3D vector containing the tonal weights. Default is DEFAULT_WEIGHTS.

    Returns
    -------
    chord : array
        Vector representation of the chord
    """
    return np.dot(ensure_norm(w), PITCH_COORDINATES[MINOR_IDXS + tonic_idx])


def major_key(tonic_idx, w=DEFAULT_WEIGHTS):
    """
    Major key representation in the spiral array space.

    Parameters
    ----------
    tonic_idx : int
        Index of the tonic of the key in NOTES_BY_FIFTHS
    w : array
        3D vector containing the tonal weights. Default is DEFAULT_WEIGHTS.

    Returns
    -------
    ce : array
        Vector representation of the center of effect of the key
    """

    chords = np.array([major_chord(tonic_idx, w),
                       major_chord(tonic_idx + 1, w),
                       major_chord(tonic_idx - 1, w)])

    return np.dot(ensure_norm(w), chords)


def minor_key(tonic_idx, w=DEFAULT_WEIGHTS, alpha=ALPHA, beta=BETA):
    """
    Minor key representation in the spiral array space

    Parameters
    ----------
    tonic_idx : int
        Index of the tonic of the key in NOTES_BY_FIFTHS
    w : array
        3D vector containing the tonal weights. Default is DEFAULT_WEIGHTS.
    alpha : float
        Preference for V vs v chord in minor key (should lie between 0 and 1)
    beta : float
        Preference for iv vs IV in minor key (should lie between 0 and 1)

    Returns
    -------
    ce : array
        Vector representation of the center of effect of the key
    """

    if alpha > 1.0 or alpha < 0:
        raise ValueError('`alpha` should be between 0 and 1.')

    if beta > 1.0 or beta < 0:
        raise ValueError('`beta` should be between 0 and 1.')

    chords = np.array([
        minor_chord(tonic_idx, w),
        (alpha * major_chord(tonic_idx + 1, w) +
         (1 - alpha) * minor_chord(tonic_idx + 1, w)),
        (beta * minor_chord(tonic_idx - 1, w) +
         (1 - beta) * major_chord(tonic_idx - 1, w))])

    return np.dot(ensure_norm(w), chords)


def cloud_diameter(cloud):
    """
    The Cloud Diameter measures the maximal tonal distance of the notes
    in a chord (or cloud of notes).

    Parameters
    ----------
    cloud : 3D array
        Array containing the coordinates in the spiral array
        of the notes in the cloud.


    Returns
    -------
    diameter : float
        Largest distance between any two notes in a cloud
    """
    return distance.pdist(cloud, metric='euclidean').max()


def center_of_effect(cloud, duration):
    """
    The center of effect condenses musical information
    in the spiral array by a single point.

    Parameters
    ----------
    cloud : 3D array
        Array containing the coordinates in the spiral array
        of the notes in the cloud.
    duration : array
        Array containing the duration of each note in the cloud


    Returns
    -------
    ce : array
       Coordinates of the center of effect
    """
    return (duration.reshape(-1, 1) * cloud).sum(0) / duration.sum()


class TonalTension(object):
    """Base class for TonalTension features"""
    def compute_tension(self, cloud, *args, **kwargs):
        raise NotImplementedError


class CloudDiameter(TonalTension):
    """
    Compute cloud diameter
    """
    def compute_tension(self, cloud, scale_factor=SCALE_FACTOR, *args, **kwargs):

        if len(cloud) > 1:
            return cloud_diameter(cloud) * scale_factor
        else:
            return 0.0


class TensileStrain(TonalTension):
    """
    Compute tensile strain
    """
    def __init__(self, tonic_idx=0, mode='major', w=DEFAULT_WEIGHTS,
                 alpha=ALPHA, beta=BETA):

        self.update_key(tonic_idx, mode, w, alpha, beta)

    def compute_tension(self, cloud, duration, scale_factor=SCALE_FACTOR,
                        *args, **kwargs):

        if duration.sum() == 0:
            return 0

        cloud_ce = center_of_effect(cloud, duration)

        return e_distance(cloud_ce, self.key_ce) * scale_factor

    def update_key(self, tonic_idx, mode, w=DEFAULT_WEIGHTS,
                   alpha=ALPHA, beta=BETA):

        if mode == 'major':
            self.key_ce = major_key(tonic_idx, w=w)
        elif mode == 'minor':
            self.key_ce = minor_key(tonic_idx, w=w,
                                    alpha=alpha, beta=beta)


class CloudMomentum(TonalTension):
    """
    Compute cloud momentum
    """
    def __init__(self):
        self.prev_ce = None

    def compute_tension(self, cloud, duration, reset=False,
                        scale_factor=SCALE_FACTOR, *args, **kwargs):

        if duration.sum() == 0:
            return 0

        if reset:
            self.prev_ce = None
        cloud_ce = center_of_effect(cloud, duration)

        if self.prev_ce is not None:
            tension = e_distance(cloud_ce, self.prev_ce) * scale_factor

        else:
            tension = 0

        self.prev_ce = cloud_ce

        return tension


def notes_to_idx(note_array):
    """
    Index of the note names in the spiral array
    """
    note_idxs = np.array([NOTES_BY_FIFTHS.index((n['step'], n['alter']))
                          for n in note_array], dtype=np.int)
    return note_idxs

def prepare_notearray(part_partgroup_list):
    """Prepare a note array with score information for computing
    tonal tension
    """
    fields = [('onset', 'f4'),
              ('duration', 'f4'),
              ('pitch', 'i4'),
              ('voice', 'i4'),
              ('id', 'U256'),
              ('step', 'U256'),
              ('alter', 'i4'),
              ('octave', 'i4'),
              ('ks_fifths', 'i4'),
              ('ks_mode', 'i4')]

    def notelist_from_part(part, pid=''):

        pnotes = part.notes_tied
        bm = part.beat_map
        km = part.key_signature_map

        on_off = np.array([bm([n.start.t, n.start.t + n.duration_tied]) for n in pnotes])
        onset = on_off[:, 0]
        duration = on_off[:, 1] - on_off[:, 0]
        pitch = np.array([n.midi_pitch for n in pnotes])
        step = np.array([n.step for n in pnotes])
        alter = np.array([n.alter if n.alter is not None else 0 for n in pnotes])
        octave = np.array([n.octave for n in pnotes])
        voice = np.array([n.voice for n in pnotes])
        key_signature = np.array([km(n.start.t) for n in pnotes])
        ids = np.array(['{0}{1}'.format(pid, n.id) for n in pnotes])

        notelist = [(o, d, p, v, i, s, a, oc, f, m)
                    for o, d, p, v, i, s, a, oc, (f, m) in zip(onset, duration, pitch,
                                                               voice, ids, step, alter,
                                                               octave, key_signature)]
        return notelist

    if isinstance(part_partgroup_list, Part):
        notelist = notelist_from_part(part, pid='')
    else:
        if isinstance(part_partgroup_list, PartGroup):
            parts = PartGroup.children
        elif isinstance(part_partgroup_list, (list, tuple)):
            parts = part_partgroup_list
        else:
            raise ValueError('`part_partgroup_list` should be a `partitura.score.Part`,'
                             ' a `partitura.score.PartGroup` or a list of '
                             '`partitura.score.Part` objetcts')
        notelist = []
        for i, part in enumerate(parts):
            notelist += notelist_from_part(part, pid='P{0:02d}_'.format(i))

    notearray = np.array(notelist, dtype=fields)

    return notearray

def estimate_tension(notearray, key, ws=1.0, ss='onset'):

    score_onset = notearray['onset']
    score_offset = score_onset + notearray['duration']

    if isinstance(ss, (float, int, np.int, np.float)):
        unique_onsets = np.arange(score_onset.min(), score_offset.max() + ws * 0.5 , step=ss)
    elif isinstance(ss, np.ndarray):
        unique_onsets = ss
    elif ss == 'onset':
        unique_onsets = np.unique(score_onset)
    else:
        raise ValueError('`ss` has to be a `float`, `int`, a numpy array or "onsets"')

    note_idxs = notes_to_idx(notearray)

    piece_coordinates = PITCH_COORDINATES[note_idxs]

    cd = CloudDiameter()
    cm = CloudMomentum()
    ts = TensileStrain(tonic_idx=NOTES_BY_FIFTHS.index(key[0]), mode=key[1])

    n_windows = len(unique_onsets)

    tonal_tension = np.zeros(n_windows,
                             dtype=[('onset', 'f4'),
                                    ('cloud_diameter', 'f4'),
                                    ('cloud_momentum', 'f4'),
                                    ('tensile_strain', 'f4')])
    tonal_tension['onset'] = unique_onsets

    for i, o in enumerate(unique_onsets):
        max_time = o + (ws * 0.5)
        min_time = o - (ws * 0.5)

        ema = set(np.where(score_offset >= max_time)[0])
        sma = set(np.where(score_onset <= max_time)[0])
        smi = set(np.where(score_onset >= min_time)[0])
        emi = set(np.where(score_offset <= max_time)[0])

        active_idx = np.array(
            list(smi.intersection(emi).union(ema.intersection(sma))),
            dtype=np.int)
        active_idx.sort()

        cloud = piece_coordinates[active_idx]
        duration = (np.minimum(max_time, score_offset[active_idx]) -
                    np.maximum(min_time, score_onset[active_idx]))

        tonal_tension['cloud_diameter'][i] = cd.compute_tension(cloud)
        tonal_tension['cloud_momentum'][i] = cm.compute_tension(cloud, duration)
        tonal_tension['tensile_strain'][i] = ts.compute_tension(cloud, duration)

    return tonal_tension

if __name__ == '__main__':
    pass









