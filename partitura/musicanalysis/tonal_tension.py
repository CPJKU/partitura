#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spiral array representation and tonal tension profiles using Herreman and
Chew's tension ribbons

References
----------

.. [1] D. Herremans and E. Chew (2016) Tension ribbons: Quantifying and
       visualising tonal tension. Proceedings of the Second International
       Conference on Technologies for Music Notation and Representation
       (TENOR), Cambridge, UK.
"""
import logging

import numpy as np
import scipy.spatial.distance as distance
from scipy.interpolate import interp1d

from partitura.utils import get_time_units_from_note_array, ensure_notearray, add_field


__all__ = ["estimate_tonaltension"]

LOGGER = logging.getLogger(__name__)

# Scaling factors
A = np.sqrt(2.0 / 15.0) * np.pi / 2.0
R = 1.0

# From Elaine Chew's thesis
DEFAULT_WEIGHTS = np.array([0.516, 0.315, 0.168])
ALPHA = 0.75
BETA = 0.75

STEPS_BY_FIFTHS = ["F", "C", "G", "D", "A", "E", "B"]

NOTES_BY_FIFTHS = []
for alt in range(-4, 5):
    NOTES_BY_FIFTHS += [(step, alt) for step in STEPS_BY_FIFTHS]

# Index of C
C_IDX = NOTES_BY_FIFTHS.index(("C", 0))
# C lies in the middle of the spiral array
T = (np.arange(len(NOTES_BY_FIFTHS)) - C_IDX) * np.pi / 2.0


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
SCALE_FACTOR = 1.0 / e_distance(
    PITCH_COORDINATES[C_IDX], PITCH_COORDINATES[NOTES_BY_FIFTHS.index(("B", 1))]
)


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

    chords = np.array(
        [
            major_chord(tonic_idx, w),
            major_chord(tonic_idx + 1, w),
            major_chord(tonic_idx - 1, w),
        ]
    )

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
        raise ValueError("`alpha` should be between 0 and 1.")

    if beta > 1.0 or beta < 0:
        raise ValueError("`beta` should be between 0 and 1.")

    chords = np.array(
        [
            minor_chord(tonic_idx, w),
            (
                alpha * major_chord(tonic_idx + 1, w)
                + (1 - alpha) * minor_chord(tonic_idx + 1, w)
            ),
            (
                beta * minor_chord(tonic_idx - 1, w)
                + (1 - beta) * major_chord(tonic_idx - 1, w)
            ),
        ]
    )

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
    return distance.pdist(cloud, metric="euclidean").max()


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

    def compute_tension(self, cloud, scale_factor=SCALE_FACTOR, **kwargs):

        if len(cloud) > 1:
            return cloud_diameter(cloud) * scale_factor
        else:
            return 0.0


class TensileStrain(TonalTension):
    """
    Compute tensile strain
    """

    def __init__(
        self, tonic_idx=0, mode="major", w=DEFAULT_WEIGHTS, alpha=ALPHA, beta=BETA
    ):

        self.update_key(tonic_idx, mode, w, alpha, beta)

    def compute_tension(
        self, cloud, duration, scale_factor=SCALE_FACTOR, *args, **kwargs
    ):

        if duration.sum() == 0:
            return 0

        cloud_ce = center_of_effect(cloud, duration)

        return e_distance(cloud_ce, self.key_ce) * scale_factor

    def update_key(self, tonic_idx, mode, w=DEFAULT_WEIGHTS, alpha=ALPHA, beta=BETA):

        if mode in ("major", None, 1):
            self.key_ce = major_key(tonic_idx, w=w)
        elif mode in ("minor", -1):
            self.key_ce = minor_key(tonic_idx, w=w, alpha=alpha, beta=beta)


class CloudMomentum(TonalTension):
    """
    Compute cloud momentum
    """

    def __init__(self):
        self.prev_ce = None

    def compute_tension(
        self, cloud, duration, reset=False, scale_factor=SCALE_FACTOR, *args, **kwargs
    ):

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
    note_idxs = np.array(
        [NOTES_BY_FIFTHS.index((n["step"], n["alter"])) for n in note_array],
        dtype=int,
    )
    return note_idxs


def prepare_note_array(note_info):

    note_array = ensure_notearray(
        note_info, include_pitch_spelling=True, include_key_signature=True
    )

    onset_unit, duration_unit = get_time_units_from_note_array(note_array)

    pitch_spelling_fields = ("step", "alter", "octave")
    key_signature_fields = ("ks_fifths", "ks_mode")

    if len(set(pitch_spelling_fields).difference(note_array.dtype.names)) > 0:
        LOGGER.info("No pitch spelling information! Estimating pitch spelling...")
        from partitura.musicanalysis.pitch_spelling import estimate_spelling

        spelling = estimate_spelling(note_array)

        note_array = add_field(note_array, spelling.dtype)

        for field in spelling.dtype.names:
            note_array[field] = spelling[field]

    if len(set(key_signature_fields).difference(note_array.dtype.names)) > 0:
        LOGGER.info("No key information! Estimating key...")
        from partitura.musicanalysis.key_identification import estimate_key
        from partitura.utils.music import key_name_to_fifths_mode, key_mode_to_int

        key_name = estimate_key(note_array)
        fifths, mode = key_name_to_fifths_mode(key_name)

        note_array = add_field(note_array, [("ks_fifths", "i4"), ("ks_mode", "i4")])

        note_array["ks_fifths"] = np.ones(len(note_array)) * fifths
        note_array["ks_mode"] = np.ones(len(note_array)) * key_mode_to_int(mode)

    return note_array


def key_map_from_keysignature(notearray, onset_unit="auto"):
    """
    Helper method to get the key map from the key signature information
    in note arrays generated with `prepare_note_array`.

    Parameters
    ----------
    notearray : structured array
        Structured array with score information. Required fields are
        `ks_fifths`, `ks_mode` and `onset`.

    Returns
    -------
    km : function
        Function that maps onset time in beats to the key signature
        in the score.
    """
    if onset_unit == "auto":
        onset_unit, _ = get_time_units_from_note_array(notearray)

    onsets = notearray[onset_unit]

    unique_onsets = np.unique(onsets)
    unique_onset_idxs = [np.where(onsets == u)[0] for u in unique_onsets]

    kss = np.zeros((len(unique_onsets), 2), dtype=int)

    for i, uix in enumerate(unique_onset_idxs):
        # Deal with potential multiple key singatures in the same onset?
        ks = np.unique(
            np.column_stack((notearray["ks_fifths"][uix], notearray["ks_mode"][uix])),
            axis=0,
        )
        if len(ks) > 1:
            LOGGER.warn(
                "Multiple Key signtures detected at score position. "
                "Taking the first one."
            )
        kss[i] = ks[0]

    return interp1d(
        unique_onsets,
        kss,
        axis=0,
        kind="previous",
        bounds_error=False,
        fill_value="extrapolate",
    )


def estimate_tonaltension(
    note_info,
    ws=1.0,
    ss="onset",
    scale_factor=SCALE_FACTOR,
    w=DEFAULT_WEIGHTS,
    alpha=ALPHA,
    beta=BETA,
):
    """
    Compute tonal tension ribbons defined in [1]_

    Parameters
    ----------
    note_info : structured array, `Part` or `PerformedPart`
        Note information as a `Part` or `PerformedPart` instances or
        as a structured array. If it is a structured array, it has to
        contain the fields generated by the `note_array` properties
        of `Part` or `PerformedPart` objects. If the array contains
        onset and duration information of both score and performance,
        (e.g., containing both `onset_beat` and `onset_sec`), the score
        information will be preferred. Furthermore, this method requires
        pitch spelling and key signature information. If a structured note
        array is provided as input, this information can be optionally
        provided in fields `step`, `alter`, `ks_fifths` and `ks_mode`.
        If these fields are not found in the input structured array,
        they will be estimated using the key and pitch spelling estimation
        methods from  `partitura.musicanalysis.estimate_key` and
        and `partitura.musicanalysis.estimate_spelling`, respectively.
    ws : {int, float, np.array}, optional
        Window size for computing the tonal tension. If a number, it determines
        the size of the window centered at each specified score position (see
        `ss` below). If a numpy array, a 2D array of shape (`len(ss)`, 2)
        specifying the left and right distance from each score position in
        `ss`. Default is 1 beat.
    ss : {float, int, np.array, 'onset'}, optional.
        Step size or score position for computing the tonal tension features.
        If a number, this parameter determines the size of the step (in beats)
        starting from the first score position. If an array, it specifies the
        score positions at which the tonal tension is estimated. If 'onset',
        it computes the tension at each unique score position (i.e., all notes
        in a chord have the same score position). Default is 'onset'.
    scale_factor : float
        A multiplicative scaling factor.
    w : np.ndarray
        Weights for the chords
    alpha : float
        Alpha.
    beta : float
        Beta.

    Returns
    -------
    tonal_tension : structured array
        Array containing the tonal tension features. It contains the fields
        `cloud_diameter`, `cloud_momentum`, `tensile_strain` and `onset`.

    References
    ----------
    .. [1] D. Herremans and E. Chew (2016) Tension ribbons: Quantifying and
           visualising tonal tension. Proceedings of the Second International
           Conference on Technologies for Music Notation and Representation
           (TENOR), Cambridge, UK.
    """

    note_array = prepare_note_array(note_info)

    onset_unit, duration_unit = get_time_units_from_note_array(note_array)

    # Open questions:
    # 1. rename score_onsets/offsets to reflect that other units are also
    # possible?
    # 2. In case the input is a performance, perhaps "cluster"/aggregate
    # the onsets into "chord" onsets or something similar?
    score_onset = note_array[onset_unit]
    score_offset = score_onset + note_array[duration_unit]

    # Determine the score position
    if isinstance(ss, (float, int)):
        unique_onsets = np.arange(
            score_onset.min(), score_offset.max() + (ss * 0.5), step=ss
        )
    elif isinstance(ss, np.ndarray):
        unique_onsets = ss
    elif ss == "onset":
        unique_onsets = np.unique(score_onset)
    else:
        raise ValueError(
            "`ss` has to be a `float`, `int`, a numpy array " 'or "onsets"'
        )

    # Determine the window sizes for each score position
    if isinstance(ws, (float, int)):
        ws = np.ones((len(unique_onsets), 2)) * 0.5 * ws
    elif isinstance(ws, np.ndarray):
        if len(ws) != len(unique_onsets):
            raise ValueError("`ws` should have the same length as " "`unique_onsets`")
    else:
        raise ValueError("`ws` has to be a `float`, `int` or a numpy array")

    note_idxs = notes_to_idx(note_array)

    # Get coordinates of the notes in the piece in the spiral array space
    piece_coordinates = PITCH_COORDINATES[note_idxs]

    # Initialize classes for computing tonal tension
    cd = CloudDiameter()
    cm = CloudMomentum()

    # Get key of the piece from key signature information
    # Perhaps add an automatic method in the future (for
    # inferring modulations?)
    km = key_map_from_keysignature(note_array, onset_unit=onset_unit)
    fifths, mode = km(unique_onsets.min()).astype(int)
    ts = TensileStrain(tonic_idx=C_IDX + fifths, mode=mode)
    # Initialize array for holding the tonal tension
    n_windows = len(unique_onsets)

    tonal_tension = np.zeros(
        n_windows,
        dtype=[
            (onset_unit, "f4"),
            ("cloud_diameter", "f4"),
            ("cloud_momentum", "f4"),
            ("tensile_strain", "f4"),
        ],
    )
    tonal_tension[onset_unit] = unique_onsets

    # Main loop for computing tension information
    for i, (o, (wlo, whi)) in enumerate(zip(unique_onsets, ws)):
        max_time = o + whi
        min_time = o - wlo

        ema = set(np.where(score_offset >= max_time)[0])
        sma = set(np.where(score_onset <= max_time)[0])
        smi = set(np.where(score_onset >= min_time)[0])
        emi = set(np.where(score_offset <= max_time)[0])

        active_idx = np.array(
            list(smi.intersection(emi).union(ema.intersection(sma))), dtype=int
        )
        active_idx.sort()

        cloud = piece_coordinates[active_idx]
        duration = np.minimum(max_time, score_offset[active_idx]) - np.maximum(
            min_time, score_onset[active_idx]
        )

        # Update key information
        if not np.all([fifths, mode] == km(o)):
            fifths, mode = km(o).astype(int)
            ts.update_key(tonic_idx=C_IDX + fifths, mode=mode)

        tonal_tension["cloud_diameter"][i] = cd.compute_tension(cloud)
        tonal_tension["cloud_momentum"][i] = cm.compute_tension(cloud, duration)
        tonal_tension["tensile_strain"][i] = ts.compute_tension(cloud, duration)

    return tonal_tension
