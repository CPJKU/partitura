#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module implement a series of mid-level descriptors of the performance expressions: asynchrony, dynamics, articulation, phrasing...
Built upon the low-level basis functions from the performance codec. 
"""

import sys
import types
from typing import Union, List
import warnings
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import least_squares
from scipy.signal import find_peaks
import numpy.lib.recfunctions as rfn
from partitura.score import ScoreLike
from partitura.performance import PerformanceLike, PerformedPart
from partitura.utils.generic import interp1d
from partitura.musicanalysis.performance_codec import to_matched_score, onsetwise_to_notewise, encode_tempo


__all__ = [
    "compute_performance_features",
]

# ordinal 
OLS = ["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff"]


class InvalidPerformanceFeatureException(Exception):
    pass

def compute_performance_features(score: ScoreLike, 
                                performance: PerformanceLike,
                                alignment: list,
                                feature_functions: Union[List, str]
):
    """
    Compute the performance features. This function is defined in the same 
    style of note_features.make_note_features

    Parameters
    ----------
    score : partitura.score.ScoreLike
        Score information, can be a part, score
    performance : partitura.performance.PerformanceLike
        Performance information, can be a ppart, performance
    alignment : list
        The score--performance alignment, a list of dictionaries
    feature_functions : list or str
        A list of performance feature functions. Elements of the list can be either
        the functions themselves or the names of a feature function as
        strings (or a mix). 
        currently implemented: 
        asynchrony_feature, articulation_feature, dynamics_feature, pedal_feature

    Returns
    -------
    performance_features : structured array
    """
    m_score, unique_onset_idxs = compute_matched_score(score, performance, alignment)

    acc = []
    if isinstance(feature_functions, str) and feature_functions == "all":
        feature_functions = list_performance_feats_functions()
    elif not isinstance(feature_functions, list):
        raise TypeError(
            "feature_functions variable {} needs to be list or 'all'".format(
                feature_functions
            )
        )

    for bf in feature_functions:
        if isinstance(bf, str):
            # get function by name from module
            func = getattr(sys.modules[__name__], bf)
        elif isinstance(bf, types.FunctionType):
            func = bf
        else:
            warnings.warn("Ignoring unknown performance feature function {}".format(bf))
        features = func(m_score, unique_onset_idxs, performance)
        # check if the size and number of the feature function are correct
        if features.size != 0:
            n_notes = len(m_score)
            if len(features) != n_notes:
                msg = (
                    "length of feature {} does not equal "
                    "number of notes {}".format(len(features), n_notes)
                )
                raise InvalidPerformanceFeatureException(msg)

            features_ = rfn.structured_to_unstructured(features)
            if np.any(np.logical_or(np.isnan(features_), np.isinf(features_))):
                problematic = np.unique(
                    np.where(np.logical_or(np.isnan(features_), np.isinf(features_)))[1]
                )
                msg = "NaNs or Infs found in the following feature: {} ".format(
                    ", ".join(np.array(features.dtype)[problematic])
                )
                raise InvalidPerformanceFeatureException(msg)

            # prefix feature names by function name
            feature_names = ["{}.{}".format(func.__name__, n) for n in features.dtype.names]
            features = rfn.rename_fields(features, dict(zip(features.dtype.names, feature_names)))

            acc.append(features)

    performance_features = rfn.merge_arrays(acc, flatten=True, usemask=False)
    return  performance_features


def compute_matched_score(score: ScoreLike, 
                            performance: PerformanceLike,
                            alignment: list,):
    """
    Compute the matched score and add the score features

    Parameters
    ----------
    score : partitura.score.ScoreLike
        Score information, can be a part, score
    performance : partitura.performance.PerformanceLike
        Performance information, can be a ppart, performance
    alignment : list
        The score--performance alignment, a list of dictionaries

    Returns
    -------
    m_score : np strutured array
    unique_onset_idxs : list
    """

    m_score, _ = to_matched_score(score, performance, alignment, include_score_markings=True)
    (time_params, unique_onset_idxs) = encode_tempo(
        score_onsets=m_score["onset"],
        performed_onsets=m_score["p_onset"],
        score_durations=m_score["duration"],
        performed_durations=m_score["p_duration"],
        return_u_onset_idx=True,
        tempo_smooth="average"
    )
    m_score = rfn.append_fields(m_score, "beat_period", time_params['beat_period'], "f4", usemask=False)

    dyn_fields = [(i, name) for i, name in enumerate(m_score.dtype.names) if "loudness" in name]
    constant_dyn = np.apply_along_axis(map_fields, 1, rfn.structured_to_unstructured(m_score), dyn_fields).flatten()

    # process the dynamcis value into discrete markings on the first beat instead of a ramp.
    constant_dyn = process_discrete_dynamics(constant_dyn)

    art_fields = [(i, name) for i, name in enumerate(m_score.dtype.names) if "articulation" in name]
    articulation = np.apply_along_axis(map_fields, 1, rfn.structured_to_unstructured(m_score), art_fields).flatten()

    m_score = rfn.rec_append_fields(m_score, "constant_dyn", constant_dyn, "U256")
    m_score = rfn.rec_append_fields(m_score, "articulation", articulation, "U256")

    return m_score, unique_onset_idxs


def list_performance_feats_functions():
    """Return a list of all feature function names defined in this module.

    The feature function names listed here can be specified by name in
    the `compute_performance_features` function. For example:

    >>> feature, names = compute_performance_features(score,
                                                    performance,
                                                    alignment,
                                                    ['asynchrony_feature'])

    Returns
    -------
    list
        A list of strings

    """
    module = sys.modules[__name__]
    bfs = []
    exclude = {"make_feature"}
    for name in dir(module):
        if name in exclude:
            continue
        member = getattr(sys.modules[__name__], name)
        if isinstance(member, types.FunctionType) and name.endswith("_feature"):
            bfs.append(name)
    return bfs


def print_performance_feats_functions():
    """Print a list of all feature function names defined in this module,
    with descriptions where available.

    """
    module = sys.modules[__name__]
    doc_indent = 4
    for name in list_performance_feats_functions():
        print("* {}".format(name))
        member = getattr(sys.modules[__name__], name)
        if member.__doc__:
            print(
                " " * doc_indent + member.__doc__.replace("\n", " " * doc_indent + "\n")
            )



# alias
list_performance_feature_functions = list_performance_feats_functions
print_performance_feature_functions = print_performance_feats_functions

def map_fields(note_info, fields):
    """
    map the one-hot fields of dynamics and articulation marking into one column with field.

    Args:
        note_info (np.array): a row slice from the note_array, without dtype names
        fields (list): list of tuples (index, field name)

    Returns:
        string: the name of the marking. 
    """

    for i, name in fields:
        if note_info[i] == 1:
            # hack for the return type
            return np.array([name.split(".")[-1]], dtype="U256")
    return np.array(["N/A"], dtype="U256")


def process_discrete_dynamics(constant_dyn):
    """reverse the continuous dynamics ramp into discrete marks on the first beat. Rest of the events are filled with N/A"""
    constant_dyn_shift = np.append(["N/A"], constant_dyn[:-1])
    positions = np.where(constant_dyn != constant_dyn_shift)[0]

    constant_dyn_ = np.full(len(constant_dyn), "N/A")
    constant_dyn_[positions] = constant_dyn[positions]

    return constant_dyn_


def parse_changing_ramp(unique_onset_idxs, m_score):
    """parse the cresceando / decresceando ramp for the actively changing subsequences. 
        Return a list of (start, end) of the changing subsequences."""

    unique_m_score = m_score[[idx[0] for idx in unique_onset_idxs]]

    increase = unique_m_score['loudness_direction_feature.loudness_incr']
    decrease = unique_m_score['loudness_direction_feature.loudness_decr']

    onset_boundaries = []
    # finding the increase & decrease boundaries 
    for ramp in [increase, decrease]:
        ramp_diff = np.append(ramp[1:], ramp[-1]) - ramp
        has_ramp_diff = ramp_diff != 0
        ramp_boundary = np.append(has_ramp_diff[0], has_ramp_diff[:-1]) ^ has_ramp_diff
        onset_boundary = unique_m_score[ramp_boundary]['onset']
        onset_boundaries.append([(onset_boundary[i], onset_boundary[i+1]) for i in range(0, len(onset_boundary), 2)])

    return tuple(onset_boundaries), unique_m_score


### Asynchrony
def asynchrony_feature(m_score: np.ndarray,
                     unique_onset_idxs: list,
                     performance: PerformanceLike,
                     v=False):
    """
    Compute the asynchrony attributes from the alignment. 

    Parameters
    ----------
    m_score : list
        correspondance between score and performance notes, with score markings. 
    unique_onset_idxs : list
        a list of arrays with the note indexes that have the same onset
    performance: PerformedPart
        The original PerformedPart object
    
    Returns
    -------
    async_ : structured array
        structured array (broadcasted to the note level) with the following fields
            delta [0, 1]: the largest time difference (in seconds) between onsets in this group 
            pitch_cor [-1, 1]: correlation between timing and pitch 
            vel_cor [-1, 1]: correlation between timing and velocity 
            voice_std [0, 1]: std of the avg timing (in seconds) of each voice in this group 
    """
    
    async_ = np.zeros(len(unique_onset_idxs), dtype=[(
        "delta", "f4"), ("pitch_cor", "f4"), ("vel_cor", "f4"), ("voice_std", "f4")])
    for i, onset_idxs in enumerate(unique_onset_idxs):

        note_group = m_score[onset_idxs]

        onset_times = note_group['p_onset']
        delta = min(onset_times.max() - onset_times.min(), 1)
        async_[i]['delta'] = delta 

        midi_pitch = note_group['pitch']
        midi_pitch = midi_pitch - midi_pitch.min() # min-scaling
        onset_times = onset_times - onset_times.min()
        cor = (-1) * np.corrcoef(midi_pitch, onset_times)[0, 1]
        # cor=nan if there is only one note in the group
        async_[i]['pitch_cor'] = (0 if np.isnan(cor) else cor)

        midi_vel = note_group['velocity'].astype(float)
        midi_vel = midi_vel - midi_vel.min()
        cor = (-1) * np.corrcoef(midi_vel, onset_times)[0, 1]
        async_[i]['vel_cor'] = (0 if np.isnan(cor) else cor)

        voices = np.unique(note_group['voice'])
        voices_onsets = []
        for voice in voices:
            note_in_voice = note_group[note_group['voice'] == voice]
            voices_onsets.append(note_in_voice['p_onset'].mean())
        async_[i]['voice_std'] = min(np.std(np.array(voices_onsets)), 1)
    
    return onsetwise_to_notewise(async_, unique_onset_idxs)


### Dynamics 
def dynamics_feature(m_score : np.ndarray,
                        unique_onset_idxs: list,
                        performance: PerformanceLike,
                        window : int = 3,
                        agg : str = "avg"):
    """
    Compute the dynamics attributes from the alignment.

    Parameters
    ----------
    m_score : list
        correspondance between score and performance notes, with score markings. 
    unique_onset_idxs : list
        a list of arrays with the note indexes that have the same onset
    performance: PerformedPart
        The original PerformedPart object
    window : int
        Length of window to look at the range of dynamic marking coverage, default 3. 
    agg : string
        how to aggregate velocity of notes with the same marking, default avg

    Returns
    -------
    dynamics_ : structured array (broadcasted to the note level) with the following fields
            agreement [-1, 1]: for each pair of dynamics, whether it agree with the OLS. Default 0
            consistency_std [0, 127]: Std of the same marking thoughout the piece. Default 0
            ramp_cor [-1, 1]: The correlation between each dynamics ramp and performed dynamics. Default 0
            tempo_cor [-1, 1]: The correlation between performed dynamics and tempo change. Default 0
    """  
    dynamics_ = np.zeros(len(m_score), dtype=[(
        "agreement", "f4"), ("consistency_std", "f4"), ("ramp_cor", "f4"), ("tempo_cor", "f4")])

    # append the marking into m_score based on the time position and windowing
    beats_with_constant_dyn = np.unique(m_score[m_score['constant_dyn'] != 'N/A']['onset'])
    markings = [m_score[m_score['onset'] == b]['constant_dyn'][0] for b in beats_with_constant_dyn]
    # TODO: windowing
    velocity = [m_score[m_score['onset'] == b]['velocity'] for b in beats_with_constant_dyn]
    velocity_agg = [np.mean(v_group) for v_group in velocity]

    constant_dynamics = list(zip(markings, velocity_agg))

    if len(constant_dynamics) < 2:
        return dynamics_

    # agreement: compare each adjacent pair of markings with their expected order, average
    marking_agreements = []
    for marking1, marking2, beat in zip(constant_dynamics, constant_dynamics[1:], beats_with_constant_dyn[1:]):
        (m1, v1), (m2, v2) = marking1, marking2
        m1_, m2_ = OLS.index(m1), OLS.index(m2)
        # preventing correlation returning nan when the values are constant
        v2 = v2 + 1e-5
        m2_ = m2_ + 1e-5        
        tau, _ = stats.kendalltau([v1, v2], [m1_, m2_])
        assert(tau == tau) # not nan
        marking_agreements.append((f"{m1}-{m2}", tau))
        dynamics_['agreement'][m_score['onset'] == beat] = tau

    # consistency: how much fluctuations does each marking have 
    markings = np.array(markings)
    velocity_agg = np.array(velocity_agg, dtype=object)
    marking_consistency = []
    for marking, beat in zip(np.unique(markings), beats_with_constant_dyn):
        marking_std = np.std(np.hstack(velocity_agg[markings == marking]))
        marking_consistency.append((f"{marking}", marking_std))
        dynamics_['consistency_std'][m_score['onset'] == beat] = marking_std


    # changing dynamics - correlation with each incr and decr ramp
    (increase_ob, decrease_ob), unique_m_score = parse_changing_ramp(unique_onset_idxs, m_score)
    for onset_boundaries, feat_name in zip([increase_ob, decrease_ob], 
                                            ['loudness_direction_feature.loudness_incr', 'loudness_direction_feature.loudness_decr']):
        for start, end in onset_boundaries:
            score_dynamics, performed_dyanmics, performed_bp = [], [], []
            for onset in unique_m_score[(unique_m_score['onset'] >= start) & (unique_m_score['onset'] <= end)]['onset']:
                score_dynamics.append(unique_m_score[unique_m_score['onset'] == onset][0][feat_name])
                performed_dyanmics.append(unique_m_score[unique_m_score['onset'] == onset]['velocity'].mean())
                performed_bp.append(unique_m_score[unique_m_score['onset'] == onset]['beat_period'].item())
            cor = stats.pearsonr(score_dynamics, performed_dyanmics)[0]
            if "decr" in feat_name:
                cor *= -1 
            ramp_mask = (m_score['onset'] >= start) & (m_score['onset'] <= end)
            dynamics_['ramp_cor'][ramp_mask] = cor
            dynamics_['tempo_cor'][ramp_mask] = (-1) * stats.pearsonr(performed_bp, performed_dyanmics)[0]

    return dynamics_


### Articulation

def articulation_feature(m_score : np.ndarray, 
                         unique_onset_idxs: list,
                         performance: PerformanceLike,
                         return_mask=False):
    """
    Compute the articulation attributes (key overlap ratio) from the alignment.
    Key overlap ratio is the ratio between key overlap time (KOT) and IOI, result in a value between (-1, inf)
        -1 is the dummy value. For normalization purposes we empirically cap the maximum to 5.
    B.Repp: Acoustics, Perception, and Production of Legato Articulation on a Digital Piano

    Parameters
    ----------
    m_score : list
        correspondance between score and performance notes, with score markings. 
    unique_onset_idxs : list
        a list of arrays with the note indexes that have the same onset
    performance: PerformedPart
        The original PerformedPart object
    return_mask : bool
        if true, return a boolean mask of legato notes, staccato notes and repeated notes

    Returns
    -------
    kor_ : structured array (1, n_notes)
        structured array on the note level with fields kor (-1, 5]
    """  
    
    m_score = rfn.append_fields(m_score, "offset", m_score['onset'] + m_score['duration'], usemask=False)
    m_score = rfn.append_fields(m_score, "p_offset", m_score['p_onset'] + m_score['p_duration'], usemask=False)

    kor_ = np.full(len(m_score), -1, dtype=[("kor", "f4")])
    if return_mask:
        mask = np.full(len(m_score), False, dtype=[("legato", "?"), ("staccato", "?"), ("repeated", "?")])

    # consider the note transition by each voice
    for voice in np.unique(m_score['voice']):
        match_voiced = m_score[m_score['voice'] == voice]
        for _, note_info in enumerate(match_voiced):

            if note_info['onset'] == match_voiced['onset'].max():  # last beat
                break
            next_note_info = get_next_note(note_info, match_voiced) # find most plausible transition

            if next_note_info: # in some cases no meaningful transition
                j = np.where(m_score == note_info)[0].item()  # original position

                if (note_info['offset'] == next_note_info['onset']):
                    kor_[j]['kor'] =  get_kor(note_info, next_note_info)

                if return_mask: # return the 
                    if (note_info['slur_feature.slur_incr'] > 0) or (note_info['slur_feature.slur_decr'] > 0): 
                        mask[j]['legato'] = True

                    if note_info['articulation'] == 'staccato':
                        mask[j]['staccato'] =  True

                    # KOR for repeated notes 
                    if (note_info['pitch'] == next_note_info['pitch']):
                        mask[j]['repeated'] =  True
    
    if return_mask:
        return kor_, mask
    else:
        return kor_


def get_kor(e1, e2):
    
    kot = e1['p_offset'] - e2['p_onset']
    ioi = e2['p_onset'] - e1['p_onset']

    kor = kot / ioi
    if kor <= -1:
        warnings.warn(f"Getting KOR smaller than -1 in {e1['onset']}-{e1['pitch']} and {e2['onset']}-{e2['pitch']}.")
    
    return min(kor, 5)

def get_next_note(note_info, match_voiced):
    """
    get the next note in the same voice that's a reasonable transition 
    note_info: the row of current note
    match_voiced: all notes in the same voice
    """

    next_position = min(o for o in match_voiced['onset'] if o > note_info['onset'])

    # if the next note is not immediate successor of the previous one...
    if next_position != note_info['onset'] + note_info['duration']:
        return None
    
    next_position_notes = match_voiced[match_voiced['onset'] == next_position]

    # from the notes in the next position, find the one that's closest pitch-wise.
    closest_idx = np.abs((next_position_notes['pitch'] - note_info['pitch'])).argmin()

    return next_position_notes[closest_idx]


### Pedals 

def pedal_feature(m_score : list, 
                unique_onset_idxs: list,
                performance: PerformanceLike):
    """
    Compute the pedal features. 

    Parameters
    ----------
    m_score : list
        correspondance between score and performance notes, with score markings. 
    unique_onset_idxs : list
        a list of arrays with the note indexes that have the same onset
    performance: PerformedPart
        The original PerformedPart object
        
    Returns
    -------
    pedal_ : structured array (4, n_notes) with fields
        onset_value [0, 127]: The interpolated pedal value at the onset
        offset_value [0, 127]: The interpolated pedal value at the onset
        to_prev_release [0, 10]: delta time from note onset to the previous pedal release 'peak'
        to_next_release [0, 10]: delta time from note offset to the next pedal release 'peak'
    """  
    
    onset_offset_pedals, ramp_func = pedal_ramp(performance.performedparts[0], m_score)

    x = np.linspace(0, 100, 200)
    y = ramp_func(x)

    peaks, _ = find_peaks(-y, prominence=10)
    peak_timepoints = x[peaks]

    release_times = np.zeros(len(m_score), dtype=[("to_prev_release", "f4"), ("to_next_release", "f4")])
    for i, note in enumerate(m_score):
        peaks_before = peak_timepoints[note['p_onset'] >= peak_timepoints]
        peaks_after = peak_timepoints[(note['p_onset'] + note['p_duration']) <= peak_timepoints]
        if len(peaks_before):
            release_times[i]["to_prev_release"] = min(note['p_onset'] - peaks_before.max(), 10)
        if len(peaks_after):
            release_times[i]["to_next_release"] = min(peaks_after.min() - (note['p_onset'] + note['p_duration']), 10)

    # plt.plot(x[peaks], y[peaks], "x")
    # plt.plot(x, y)
    # plt.show()

    return rfn.merge_arrays([onset_offset_pedals, release_times], flatten=True, usemask=False)

def pedal_ramp(ppart: PerformedPart,
               m_score: np.ndarray):
    """Pedal ramp in the same shape as the m_score.

    Returns:
    * pramp : a ramp function that ranges from 0
                  to 127 with the change of sustain pedal
    """
    pedal_controls = ppart.controls
    W = np.zeros((len(m_score), 2))
    onset_timepoints = m_score['p_onset']
    offset_timepoints = m_score['p_onset'] + m_score['p_duration']

    timepoints = [control['time'] for control in pedal_controls]
    values = [control['value'] for control in pedal_controls]

    agg_ramp_func = interp1d(timepoints, values, bounds_error=False, fill_value=0)
    W[:, 0] = agg_ramp_func(onset_timepoints)
    W[:, 1] = agg_ramp_func(offset_timepoints)

    # Filter out NaN values
    W[np.isnan(W)] = 0.0

    return np.array([tuple(i) for i in W], dtype=[("onset_value", "f4"), ("offset_value", "f4")]), agg_ramp_func



### Phrasing


def freiberg_kinematic(params, xdata, ydata):
    w, q = params
    return ydata - (1 + (w ** q - 1) * xdata) ** (1/q)


def get_phrase_end(m_score, unique_onset_idxs):
    """
    Returns a list of possible phrase endings for analyzing slowing down structure. 
    (current implementation only takes last 4 beats, need for more advanced segmentation algorithm.)

    Parameters
    ----------
    m_score : structured array
        correspondance between score and performance notes, with score markings. 
    unique_onset_idxs : list
        a list of arrays with the note indexes that have the same onset

    Returns
    -------
    endings : list
        list of tuples with (beats, tempo)
    """
    beat_first_note =  [group[0] for group in unique_onset_idxs]
    m_score_beats = m_score[beat_first_note]

    # last 4 beats
    final_beat = m_score_beats['onset'][-1]
    prase_ending = m_score_beats[(m_score_beats['onset'] >= final_beat - 4)]
    xdata, ydata = prase_ending['onset'], 60 / prase_ending['beat_period']

    endings = [(xdata, ydata)]
    return endings


def phrasing_attributes(m_score, unique_onset_idxs, plot=False):
    """
    Unfinished! after finishing will update to phrasing_feature
    rubato:
        Model the final tempo curve (last 2 measures) using Friberg & Sundberg’s kinematic model: 
            (https://www.researchgate.net/publication/220723460_Evidence_for_Pianist-specific_Rubato_Style_in_Chopin_Nocturnes)
        v(x) = (1 + (w^q − 1) * x)^(1/q), 
        w: the final tempo (normalized between 0 and 1, assuming normalized )
        q: variation in curvature

    Parameters
    ----------
    m_score : structured array
        correspondance between score and performance notes, with score markings. 
    unique_onset_idxs : list
        a list of arrays with the note indexes that have the same onset
        
    Returns
    -------
    pharsing_ : structured array
        structured array on the (phrase?) level with fields w and q. 
    """

    endings = get_phrase_end(m_score, unique_onset_idxs)
    phrasing_ = np.zeros(len(endings), dtype=[("rubato_w", "f4"), ("rubato_q", "f4")])     

    for i, ending in enumerate(endings):
        xdata, ydata = ending

        # normalize x and y. y: initial tempo as 1
        xdata = (xdata - xdata.min()) * (1 / (xdata.max() - xdata.min()))
        ydata = (ydata - 0) * (1 / (ydata.max() - 0))

        params_init = np.array([0.5, -1])
        res = least_squares(freiberg_kinematic, params_init, args=(xdata, ydata))
        
        w, q = res.x
        phrasing_[i]['rubato_w'] = w
        phrasing_[i]['rubato_q'] = q

        if plot:
            plt.scatter(xdata, ydata, marker="+", c="red")
            xline = np.linspace(0, 1, 100)
            plt.plot(xline, (1 + (w ** q - 1) * xline) ** (1/q))
            plt.ylim(0, 1.2)
            plt.title(f"Friberg & Sundberg kinematic rubato curve with w={round(w, 2)} and q={round(q, 2)}")
            plt.show()

    return phrasing_
