import numpy as np
import numpy.lib.recfunctions as rfn
try:
    import torch
except ImportError:
    # Dummy module to avoid ImportErrors
    class DummyTorch(object):
        Tensor = np.ndarray

        def __init__(self):
            pass

    torch = DummyTorch()


from partitura.score import Part
from partitura.performance import PerformedPart
from scipy.interpolate import interp1d

__all__ = ["encode_performance", "decode_performance"]


def encode_performance(
    part: Part,
    ppart: PerformedPart,
    alignment: list,
    return_u_onset_idx=False,
):
    """
    Encode expressive parameters from a matched performance


    Parameters
    ----------
    part : partitura.Part
        The score full either load from a match file.
    ppart : partitura.PerformedPart
        A list of Dictionaries
    alignment : list
        The score--performance alignment, a list of dictionaries
    return_u_onset_idx : bool
        Return the indices of the unique score onsets

    Returns
    -------
    parameters : structured array
        A performance array with 4 fields: beat_period, velocity,
        timing, and rticulation_log.
    snote_ids : dict
        A dict of snote_ids corresponding to performance notes.
    unique_onset_idxs : list (optional)
        List of unique onset ids. Returned only when return_u_onset_idx
        is set to True.
    """

    m_score, snote_ids = to_matched_score(part, ppart, alignment)

    # Get time-related parameters
    (time_params, unique_onset_idxs) = encode_tempo(
        score_onsets=m_score["onset"],
        performed_onsets=m_score["p_onset"],
        score_durations=m_score["duration"],
        performed_durations=m_score["p_duration"],
        return_u_onset_idx=True,
    )

    # Get dynamics-related parameters
    dynamics_params = np.array(m_score["velocity"] / 127.0, dtype=[("velocity", "f4")])

    # Fixing random error
    parameters = time_params
    parameters["velocity"] = dynamics_params["velocity"]

    if return_u_onset_idx:
        return parameters, snote_ids, unique_onset_idxs
    else:
        return parameters, snote_ids


def decode_performance(
    part: Part,
    performance_array: np.ndarray,
    snote_ids=None,
    part_id=None,
    part_name=None,
    return_alignment=False,
    *args,
    **kwargs
) -> PerformedPart:
    """
    Given a Part (score) and a performance array return a PerformedPart.

    Parameters
    ----------
    part : partitura.score.Part
        A partitura Part representing a score
    performance_array : structured array
        A performed array related to the part.
    snote_ids : list
    part_id : str
    part_name : str
    return_alignment : bool
        True returns alignment list of dicts.

    Returns
    -------
    ppart : partitura.performance.PerformedPart
        A partitura PerformedPart.
    alignment: list (optional)
        A list of dicts for the alignment.
    """

    snotes = part.notes_tied

    if snote_ids is None:
        snote_ids = [n.id for n in snotes]

    # sort
    snote_dict = dict((n.id, n) for n in snotes)
    snote_info = np.array(
        [
            (
                snote_dict[i].midi_pitch,
                snote_dict[i].start.t,
                snote_dict[i].start.t + snote_dict[i].duration_tied,
            )
            for i in snote_ids
        ],
        dtype=[("pitch", "i4"), ("onset", "f4"), ("offset", "f4")],
    )
    sort_idx = np.lexsort((snote_info["pitch"], snote_info["onset"]))

    bm = part.beat_map
    onsets = bm(snote_info["onset"])[sort_idx]
    durations = (bm(snote_info["offset"]) - bm(snote_info["onset"]))[sort_idx]
    pitches = snote_info["pitch"][sort_idx]

    pitches = np.clip(pitches, 1, 127)

    dynamics_params = performance_array["velocity"][sort_idx]
    time_params = performance_array[
        list(("beat_period", "timing", "articulation_log"))
    ][sort_idx]

    onsets_durations = decode_time(
        score_onsets=onsets,
        score_durations=durations,
        parameters=time_params,
        *args,
        **kwargs
    )

    velocities = np.round(dynamics_params * 127.0)

    velocities = np.clip(velocities, 1, 127)

    notes = []
    for nid, (onset, duration), velocity, pitch in zip(
        snote_ids, onsets_durations, velocities, pitches
    ):
        notes.append(
            dict(
                id=nid,
                midi_pitch=int(pitch),
                note_on=onset,
                note_off=onset + duration,
                sound_off=onset + duration,
                velocity=int(velocity),
            )
        )
    # * rescale according to default values?
    ppart = PerformedPart(id=part_id, part_name=part_name, notes=notes)

    if return_alignment:
        alignment = []
        for snote, pnote in zip(part.notes_tied, ppart.notes):
            alignment.append(
                dict(label="match", score_id=snote.id, performance_id=pnote["id"])
            )

        return ppart, alignment
    else:

        return ppart


def decode_time(score_onsets, score_durations, parameters, *args, **kwargs):
    """
    Decode a performance into onsets and durations in seconds
    for each note in the score.
    """
    score_onsets = score_onsets.astype(float, copy=False)
    score_durations = score_durations.astype(float, copy=False)

    score_info = get_unique_seq(
        onsets=score_onsets,
        offsets=score_onsets + score_durations,
        unique_onset_idxs=None,
        return_diff=True,
    )
    unique_onset_idxs = score_info["unique_onset_idxs"]
    diff_u_onset_score = score_info["diff_u_onset"]

    time_param = np.array(
        [tuple([np.mean(parameters["beat_period"][uix])]) for uix in unique_onset_idxs],
        dtype=[("beat_period", "f4")],
    )

    beat_period = time_param["beat_period"]

    ioi_perf = diff_u_onset_score * beat_period

    eq_onset = np.cumsum(np.r_[0, ioi_perf])

    performance = np.zeros((len(score_onsets), 2))

    for i, jj in enumerate(unique_onset_idxs):
        # decode onset
        performance[jj, 0] = eq_onset[i] - parameters["timing"][jj]
        # decode duration
        performance[jj, 1] = decode_articulation(
            score_durations=score_durations[jj],
            articulation_parameter=parameters["articulation_log"][jj],
            beat_period=beat_period[i],
        )

    performance[:, 0] -= np.min(performance[:, 0])

    return performance


def decode_articulation(score_durations, articulation_parameter, beat_period):
    """
    Decode articulation
    """
    art_ratio = 2 ** articulation_parameter
    dur = art_ratio * score_durations * beat_period

    return dur


def encode_tempo(
    score_onsets: np.ndarray,
    performed_onsets: np.ndarray,
    score_durations,
    performed_durations,
    return_u_onset_idx: bool = False,
) -> np.ndarray:
    """
    Compute time-related performance parameters from a performance
    """
    if score_onsets.shape != performed_onsets.shape:
        raise ValueError("The performance and the score should be of " "the same size")

    # use float64, float32 led to problems that x == x + eps evaluated to True
    # Maybe replace by np.isclose
    score_onsets = score_onsets.astype(float, copy=False)
    performed_onsets = performed_onsets.astype(float, copy=False)
    score_durations = score_durations.astype(float, copy=False)
    performed_durations = performed_durations.astype(float, copy=False)
    score = np.column_stack((score_onsets, score_durations))
    performance = np.column_stack((performed_onsets, performed_durations))

    # Compute beat period
    beat_period, s_onsets, unique_onset_idxs = tempo_by_average(
        score_onsets=score[:, 0],
        performed_onsets=performance[:, 0],
        score_durations=score[:, 1],
        performed_durations=performance[:, 1],
        return_onset_idxs=True,
    )

    # Compute equivalent onsets
    eq_onsets = (
        np.cumsum(np.r_[0, beat_period[:-1] * np.diff(s_onsets)])
        + performance[unique_onset_idxs[0], 0].mean()
    )

    # Compute tempo parameter
    # TODO fix normalization
    tempo_params = [beat_period]

    # Compute articulation parameter
    articulation_param = encode_articulation(
        score_durations=score[:, 1],
        performed_durations=performance[:, 1],
        unique_onset_idxs=unique_onset_idxs,
        beat_period=beat_period,
    )

    # Initialize array of parameters
    parameter_names = ["beat_period", "velocity", "timing", "articulation_log"]
    parameters = np.zeros(len(score), dtype=[(pn, "f4") for pn in parameter_names])
    parameters["articulation_log"] = articulation_param
    for i, jj in enumerate(unique_onset_idxs):
        parameters["beat_period"][jj] = tempo_params[0][i]
        # Defined as in Eq. (3.9) in Thesis (pp. 34)
        parameters["timing"][jj] = eq_onsets[i] - performance[jj, 0]

    if return_u_onset_idx:
        return parameters, unique_onset_idxs
    else:
        return parameters


def tempo_by_average(
    score_onsets,
    performed_onsets,
    score_durations,
    performed_durations,
    unique_onset_idxs=None,
    input_onsets=None,
    return_onset_idxs=False,
):
    """
    Computes a tempo curve using the average of the onset times of all
    notes belonging to the same score onset.
    Parameters
    ----------
    score_onsets : np.ndarray
        Onset in beats of each note in the score.
    performed_onsets : np.ndarray
        Performed onsets in seconds of each note in the score.
    score_durations : np.ndarray
        Duration in beats of each note in the score.
    performed_durations : np.ndarray
        Performed duration in seconds of each note in the score.
    unique_onset_idxs : np.ndarray or None (optional)
        Indices of the notes with the same score onset. (By default is None,
        and is therefore, inferred from `score_onsets`).
    input_onsets : np.ndarray or None
        Input onset times in beats at which the tempo curve is to be
        sampled (by default is None, which means that the tempo curve
        is returned for each unique score onset)
    return_onset_idxs : bool
        Return the indices of the unique score onsets (Default is False)
    Returns
    -------
    tempo_curve : np.ndarray
        Tempo curve in seconds per beat (spb). If `input_onsets` was provided,
        this array contains the value of the tempo in spb for each onset
        in `input_onsets`. Otherwise, this array contains the value of the
        tempo in spb for each unique score onset.
    input_onsets : np.ndarray
        The score onsets corresponding to each value of the tempo curve.
    unique_onset_idxs: list
        Each element of the list is an array of the indices of the score
        corresponding to the elements in `tempo_curve`. Only returned if
        `return_onset_idxs` is True.
    """
    # use float64, float32 led to problems that x == x + eps evaluated
    # to True
    score_onsets = np.array(score_onsets).astype(float, copy=False)
    performed_onsets = np.array(performed_onsets).astype(float, copy=False)
    score_durations = np.array(score_durations).astype(float, copy=False)
    performed_durations = np.array(performed_durations).astype(float, copy=False)

    # Get unique onsets if no provided
    if unique_onset_idxs is None:
        # Get indices of the unique onsets (quantize score onsets)
        unique_onset_idxs = get_unique_onset_idxs((1e4 * score_onsets).astype(int))

    # Get score information
    score_info = get_unique_seq(
        onsets=score_onsets,
        offsets=score_onsets + score_durations,
        unique_onset_idxs=unique_onset_idxs,
    )
    # Get performance information
    perf_info = get_unique_seq(
        onsets=performed_onsets,
        offsets=performed_onsets + performed_durations,
        unique_onset_idxs=unique_onset_idxs,
    )

    # unique score onsets
    unique_s_onsets = score_info["u_onset"]
    # equivalent onsets
    eq_onsets = perf_info["u_onset"]

    # Monotonize times
    eq_onset_mt, unique_s_onsets_mt = monotonize_times(
        eq_onsets, deltas=unique_s_onsets
    )

    # Estimate Beat Period
    perf_iois = np.diff(eq_onset_mt)
    s_iois = np.diff(unique_s_onsets_mt)
    beat_period = perf_iois / s_iois

    tempo_fun = interp1d(
        unique_s_onsets_mt[:-1],
        beat_period,
        kind="zero",
        bounds_error=False,
        fill_value=(beat_period[0], beat_period[-1]),
    )

    if input_onsets is None:
        input_onsets = unique_s_onsets[:-1]

    tempo_curve = tempo_fun(input_onsets)

    if return_onset_idxs:
        return tempo_curve, input_onsets, unique_onset_idxs
    else:
        return tempo_curve, input_onsets


def get_unique_seq(onsets, offsets, unique_onset_idxs=None, return_diff=False):
    """
    Get unique onsets of a sequence of notes
    """
    eps = np.finfo(float).eps

    first_time = np.min(onsets)

    # ensure last score time is later than last onset
    last_time = max(np.max(onsets) + eps, np.max(offsets))

    total_dur = last_time - first_time

    if unique_onset_idxs is None:
        # unique_onset_idxs = unique_onset_idx(score[:, 0])
        unique_onset_idxs = get_unique_onset_idxs(onsets)

    u_onset = np.array([np.mean(onsets[uix]) for uix in unique_onset_idxs])
    # add last offset, so we have as many IOIs as notes
    u_onset = np.r_[u_onset, last_time]

    output_dict = dict(
        u_onset=u_onset, total_dur=total_dur, unique_onset_idxs=unique_onset_idxs
    )

    if return_diff:
        output_dict["diff_u_onset"] = np.diff(u_onset)

    return output_dict


def get_unique_onset_idxs(
    onsets, eps: float = 1e-6, return_unique_onsets: bool = False
):
    """
    Get unique onsets and their indices.
    Parameters
    ----------
    onsets : np.ndarray
        Score onsets in beats.
    eps : float
        Small epsilon (for dealing with quantization in symbolic scores).
        This is particularly useful for dealing with triplets and other
        similar rhytmical structures that do not have a finite decimal
        representation.
    return_unique_onsets : bool (optional)
        If `True`, returns the unique score onsets.
    Returns
    -------
    unique_onset_idxs : np.ndarray
        Indices of the unique onsets in the score.
    unique_onsets : np.ndarray
        Unique score onsets
    """
    # Do not assume that the onsets are sorted
    # (use a stable sorting algorithm for preserving the order
    # of elements with the same onset, which is useful e.g. if the
    # notes within a same onset are ordered by pitch)
    sort_idx = np.argsort(onsets, kind="mergesort")
    split_idx = np.where(np.diff(onsets[sort_idx]) > eps)[0] + 1
    unique_onset_idxs = np.split(sort_idx, split_idx)

    if return_unique_onsets:
        # Instead of np.unique(onsets)
        unique_onsets = np.array([onsets[uix].mean() for uix in unique_onset_idxs])

        return unique_onset_idxs, unique_onsets
    else:
        return unique_onset_idxs


def to_matched_score(part, ppart, alignment):
    # remove repetitions from aligment note ids
    for a in alignment:
        if a["label"] == "match":
            # FOR MAGALOFF/ZEILINGER DO SPLIT:
            # a['score_id'] = str(a['score_id']).split('-')[0]
            a["score_id"] = str(a["score_id"])

    part_by_id = dict((n.id, n) for n in part.notes_tied)
    ppart_by_id = dict((n["id"], n) for n in ppart.notes)

    # pair matched score and performance notes

    note_pairs = [
        (part_by_id[a["score_id"]], ppart_by_id[a["performance_id"]])
        for a in alignment
        if (a["label"] == "match" and a["score_id"] in part_by_id)
    ]

    ms = []
    # sort according to onset (primary) and pitch (secondary)
    pitch_onset = [(sn.midi_pitch, sn.start.t) for sn, _ in note_pairs]
    sort_order = np.lexsort(list(zip(*pitch_onset)))
    beat_map = part.beat_map
    snote_ids = []
    for i in sort_order:

        sn, n = note_pairs[i]
        sn_on, sn_off = beat_map([sn.start.t, sn.start.t + sn.duration_tied])
        sn_dur = sn_off - sn_on
        # hack for notes with negative durations
        n_dur = max(n["sound_off"] - n["note_on"], 60 / 200 * 0.25)
        ms.append((sn_on, sn_dur, sn.midi_pitch, n["note_on"], n_dur, n["velocity"]))
        snote_ids.append(sn.id)

    fields = [
        ("onset", "f4"),
        ("duration", "f4"),
        ("pitch", "i4"),
        ("p_onset", "f4"),
        ("p_duration", "f4"),
        ("velocity", "i4"),
    ]

    return np.array(ms, dtype=fields), snote_ids


def encode_articulation(
    score_durations, performed_durations, unique_onset_idxs, beat_period
):
    """
    Encode articulation
    """
    articulation = np.zeros_like(score_durations)
    for idx, bp in zip(unique_onset_idxs, beat_period):

        sd = score_durations[idx]
        pd = performed_durations[idx]

        # indices of notes with duration 0 (grace notes)
        grace_mask = sd <= 0

        # Grace notes have an articulation ratio of 1
        sd[grace_mask] = 1
        pd[grace_mask] = bp
        # Compute log articulation ratio
        articulation[idx] = np.log2(pd / (bp * sd))

    return articulation


def monotonize_times(s, deltas=None):
    """Interpolate linearly over as many points in `s` as necessary to
    obtain a monotonic sequence. The minimum and maximum of `s` are
    prepended and appended, respectively, to ensure monotonicity at
    the bounds of `s`.
    Parameters
    ----------
    s : ndarray
        a sequence of numbers
    strict : bool
        when True, return a strictly monotonic sequence (default: True)
    Returns
    -------
    ndarray
       a monotonic sequence that has been linearly interpolated using a subset of s
    """
    eps = np.finfo(float).eps

    _s = np.r_[np.min(s) - eps, s, np.max(s) + eps]
    if deltas is not None:
        _deltas = np.r_[np.min(deltas) - eps, deltas, np.max(deltas) + eps]
    else:
        _deltas = None
    mask = np.ones(_s.shape[0], dtype=bool)
    mask[0] = mask[-1] = False
    idx = np.arange(_s.shape[0])
    s_mono = interp1d(idx[mask], _s[mask])(idx[1:-1])
    return _s[mask], _deltas[mask]


def notewise_to_onsetwise(notewise_inputs, unique_onset_idxs):
    """Agregate basis functions per onset"""
    if isinstance(notewise_inputs, np.ndarray):
        if notewise_inputs.ndim == 1:
            shape = len(unique_onset_idxs)
        else:
            shape = (len(unique_onset_idxs),) + notewise_inputs.shape[1:]
        onsetwise_inputs = np.zeros(shape, dtype=notewise_inputs.dtype)
    elif isinstance(notewise_inputs, torch.Tensor):
        onsetwise_inputs = torch.zeros(
            (len(unique_onset_idxs), notewise_inputs.shape[1]),
            dtype=notewise_inputs.dtype,
        )

    for i, uix in enumerate(unique_onset_idxs):
        try:
            onsetwise_inputs[i] = notewise_inputs[uix].mean(0)
        except TypeError:
            for tn in notewise_inputs.dtype.names:
                onsetwise_inputs[i][tn] = notewise_inputs[uix][tn].mean()
    return onsetwise_inputs


def onsetwise_to_notewise(onsetwise_input, unique_onset_idxs):
    """Expand onsetwise predictions for each note"""
    n_notes = sum([len(uix) for uix in unique_onset_idxs])
    if isinstance(onsetwise_input, np.ndarray):
        if onsetwise_input.ndim == 1:
            shape = n_notes
        else:
            shape = (n_notes,) + onsetwise_input.shape[1:]
        notewise_inputs = np.zeros(shape, dtype=onsetwise_input.dtype)
    elif isinstance(onsetwise_input, torch.Tensor):
        notewise_inputs = torch.zeros(n_notes, dtype=onsetwise_input.dtype)
    for i, uix in enumerate(unique_onset_idxs):
        notewise_inputs[uix] = onsetwise_input[[i]]
    return notewise_inputs
