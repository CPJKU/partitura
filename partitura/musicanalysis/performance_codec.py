import numpy.lib.recfunctions as rfn
import numpy as np
from scipy.interpolate import interp1d

class PerformanceCodec(object):
    """
    A Codec for decomposing a performance into expressive parameters
    """

    def __init__(self):
        super().__init__()
        self.parameter_names = ['beat_period', 'velocity', 'timing', 'articulation_log']

    def encode(self, part, ppart, alignment, return_u_onset_idx=False):
        """
        Encode expressive parameters from a matched performance
        Parameters
        ----------
        matched_score : structured array
            Performance matched to its score.
        snote_ids : list
            NoteID's of the matched notes in the performance.
        return_u_onset_idx : bool
            Return the indices of the unique score onsets
        """

        m_score, snote_ids = to_matched_score(part, ppart, alignment)

        # Get time-related parameters
        (time_params,
         unique_onset_idxs) = self.tempo_encode(
            score_onsets=m_score['onset'],
            performed_onsets=m_score['p_onset'],
            score_durations=m_score['duration'],
            performed_durations=m_score['p_duration'],
            return_u_onset_idx=True)

        # Get dynamics-related parameters
        dynamics_params = np.array(m_score['velocity'] / 127.0, dtype=[('velocity', 'f4')])
        # Concatenate parameters into a single matrix
        parameters = rfn.merge_arrays([time_params, dynamics_params],
                                      flatten=True, usemask=False)

        if return_u_onset_idx:
            return parameters, snote_ids, unique_onset_idxs
        else:
            return parameters, snote_ids

    def tempo_encode(self, score_onsets, performed_onsets,
               score_durations, performed_durations,
               return_u_onset_idx=False):
        """
        Compute time-related performance parameters from a performance
        """
        if score_onsets.shape != performed_onsets.shape:
            raise ValueError('The performance and the score should be of '
                             'the same size')

        # use float64, float32 led to problems that x == x + eps
        # evaluated to True
        score_onsets = score_onsets.astype(np.float64, copy=False)
        performed_onsets = performed_onsets.astype(np.float64, copy=False)
        score_durations = score_durations.astype(np.float64, copy=False)
        performed_durations = performed_durations.astype(np.float64, copy=False)
        score = np.column_stack((score_onsets, score_durations))
        performance = np.column_stack((performed_onsets, performed_durations))

        # Compute beat period
        beat_period, s_onsets, unique_onset_idxs = self.tempo_by_average(
            score_onsets=score[:, 0],
            performed_onsets=performance[:, 0],
            score_durations=score[:, 1],
            performed_durations=performance[:, 1],
            return_onset_idxs=True)

        # Compute equivalent onsets
        eq_onsets = (np.cumsum(np.r_[0, beat_period[:-1] * np.diff(s_onsets)]) +
                     performance[unique_onset_idxs[0], 0].mean())

        # Compute tempo parameter
        # TODO fix normalization
        tempo_params = self.normalization['scale'](beat_period)

        # Compute articulation parameter
        articulation_param = encode_articulation(
            score_durations=score[:, 1],
            performed_durations=performance[:, 1],
            unique_onset_idxs=unique_onset_idxs,
            beat_period=beat_period)

        # Initialize array of parameters
        parameters = np.zeros(len(score),
                              dtype=[(pn, 'f4') for pn in self.parameter_names])
        parameters['articulation_log'] = articulation_param
        for i, jj in enumerate(unique_onset_idxs):
            parameters[self.normalization['param_names'][0]][jj] = tempo_params[0][i]
            # Defined as in Eq. (3.9) in Thesis (pp. 34)
            parameters['timing'][jj] = eq_onsets[i] - performance[jj, 0]

            for pn, tp in zip(self.normalization['param_names'][1:], tempo_params[1:]):
                parameters[pn][jj] = tp[i]

        if return_u_onset_idx:
            return parameters, unique_onset_idxs
        else:
            return parameters

    def tempo_by_average(self, score_onsets, performed_onsets,
                         score_durations, performed_durations,
                         unique_onset_idxs=None,
                         input_onsets=None,
                         return_onset_idxs=False):
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
        score_onsets = np.array(score_onsets).astype(np.float64, copy=False)
        performed_onsets = np.array(performed_onsets).astype(np.float64, copy=False)
        score_durations = np.array(score_durations).astype(np.float64, copy=False)
        performed_durations = np.array(performed_durations).astype(np.float64, copy=False)

        # Get unique onsets if no provided
        if unique_onset_idxs is None:
            # Get indices of the unique onsets (quantize score onsets)
            unique_onset_idxs = get_unique_onset_idxs((1e4 * score_onsets).astype(np.int))

        # Get score information
        score_info = get_unique_seq(onsets=score_onsets,
                                    offsets=score_onsets + score_durations,
                                    unique_onset_idxs=unique_onset_idxs,
                                    return_diff=False)
        # Get performance information
        perf_info = get_unique_seq(onsets=performed_onsets,
                                   offsets=performed_onsets + performed_durations,
                                   unique_onset_idxs=unique_onset_idxs,
                                   return_diff=False)

        # unique score onsets
        unique_s_onsets = score_info['u_onset']
        # equivalent onsets
        eq_onsets = perf_info['u_onset']

        # Monotonize times
        eq_onset_mt, unique_s_onsets_mt = monotonize_times(eq_onsets,
                                                           deltas=unique_s_onsets)

        # Estimate Beat Period
        perf_iois = np.diff(eq_onset_mt)
        s_iois = np.diff(unique_s_onsets_mt)
        beat_period = perf_iois / s_iois

        tempo_fun = interp1d(unique_s_onsets_mt[:-1], beat_period, kind='zero',
                             bounds_error=False,
                             fill_value=(beat_period[0], beat_period[-1]))

        if input_onsets is None:
            input_onsets = unique_s_onsets[:-1]

        tempo_curve = tempo_fun(input_onsets)

        if return_onset_idxs:
            return tempo_curve, input_onsets, unique_onset_idxs
        else:
            return tempo_curve, input_onsets

def get_unique_seq(onsets, offsets, unique_onset_idxs=None,
                   return_diff=False):
    """
    Get unique onsets of a sequence of notes
    """
    eps = np.finfo(np.float32).eps

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
        u_onset=u_onset,
        total_dur=total_dur,
        unique_onset_idxs=unique_onset_idxs)

    if return_diff:
        output_dict['diff_u_onset'] = np.diff(u_onset)

    return output_dict

def get_unique_onset_idxs(onsets, eps=1e-6, return_unique_onsets=False):
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
    sort_idx = np.argsort(onsets, kind='mergesort')
    split_idx = np.where(np.diff(onsets[sort_idx]) > eps)[0] + 1
    unique_onset_idxs = np.split(sort_idx, split_idx)

    if return_unique_onsets:
        # Instead of np.unique(onsets)
        unique_onsets = np.array([onsets[uix].mean()
                                  for uix in unique_onset_idxs])

        return unique_onset_idxs, unique_onsets
    else:
        return unique_onset_idxs

def to_matched_score(part, ppart, alignment):
    # remove repetitions from aligment note ids
    for a in alignment:
        if a['label'] == 'match':
            # FOR MAGALOFF/ZEILINGER DO SPLIT:
            # a['score_id'] = str(a['score_id']).split('-')[0]
            a['score_id'] = str(a['score_id'])

    part_by_id = dict((n.id, n) for n in part.notes_tied)
    ppart_by_id = dict((n['id'], n) for n in ppart.notes)

    # pair matched score and performance notes

    note_pairs = [(part_by_id[a['score_id']],
                   ppart_by_id[a['performance_id']])
                  for a in alignment if (a['label'] == 'match' and
                                         a['score_id'] in part_by_id)]

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
        n_dur = max(n['sound_off'] - n['note_on'], 60 / 200 * 0.25)
        ms.append((sn_on, sn_dur, sn.midi_pitch, n['note_on'], n_dur, n['velocity']))
        snote_ids.append(sn.id)

    fields = [('onset', 'f4'), ('duration', 'f4'), ('pitch', 'i4'),
              ('p_onset', 'f4'), ('p_duration', 'f4'), ('velocity', 'i4')]

    return np.array(ms, dtype=fields), snote_ids

def encode_articulation(score_durations, performed_durations,
                        unique_onset_idxs, beat_period):
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

    # articulation[np.where(np.isnan(articulation))] = 0
    if len(np.where(np.isnan(articulation))[0]) != 0:
        import pdb
        pdb.set_trace()

    return articulation

def monotonize_times(s, strict=True, deltas=None, return_interp_seq=False):
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
    if strict:
        eps = np.finfo(np.float32).eps
    else:
        eps = 1e-3

    _s = np.r_[np.min(s) - eps, s, np.max(s) + eps]
    if deltas is not None:
        _deltas = np.r_[np.min(deltas) - eps, deltas, np.max(deltas) + eps]
    else:
        _deltas = None
    mask = np.ones(_s.shape[0], dtype=np.bool)
    monotonic_mask = _monotonize_times(_s, mask, strict, _deltas)[1:-1]  # is the mask always changed in place?
    mask[0] = mask[-1] = False
    # s_mono = interp1d(idx[mask], _s[mask])(idx[1:-1])
    if return_interp_seq:
        idx = np.arange(_s.shape[0])
        s_mono = interp1d(idx[mask], _s[mask])(idx[1:-1])
        return s_mono
    else:
        return _s[mask], _deltas[mask]

def _monotonize_times(s, mask, strict, deltas=None):
    """
    Return a mask such that s[mask] is monotonic.
    """

    # print('is monotonic?')
    if _is_monotonic(s, mask, strict, deltas):
        # print('yes')
        return mask
    else:
        # print('no')
        p = _select_problematic(s, mask, deltas)
        # print('select problematic', p)
        mask[p] = False

        return _monotonize_times(s, mask, strict, deltas)



def _select_problematic(s, mask, deltas=None):
    """Return the index i of an element of s, such that removing s[i]
    reduces the non-monotonicity of s[mask]. This method assumes the
    first and last element of `s` are the the minimum and maximum,
    respectively.
    Parameters
    ----------
    s : ndarray
        a sequence of numbers
    mask : ndarray(bool)
        a boolean mask to be applied to s
    Returns
    -------
    int
       an index i such that removing s[i] from s reduces
       non-monotonicity
    """
    if deltas is not None:
        diffs = np.diff(s[mask]) / np.diff(deltas[mask])
    else:
        diffs = np.diff(s[mask])
    # ignore first and last elements
    problematic = np.argmin(diffs[1:] * diffs[:-1]) + 1
    return np.where(mask)[0][problematic]

def _is_monotonic(s, mask, strict=True, deltas=None):
    """Return True when `s[mask]` is monotonic. When `strict` is True,
    check for strict monotonicity.
    Parameters
    ----------
    s : ndarray
        a sequence of numbers
    mask : ndarray(bool)
        a boolean mask to be applied to `s`
    strict : bool
        when True, return a strictly monotonic sequence (default: True)
    """

    if s[mask].shape[0] < 4:
        return True
    else:
        if strict:
            return np.all(np.diff(s[mask]) / np.diff(deltas[mask]) > 0.0)
        else:
            return np.all(np.diff(s[mask]) / np.diff(deltas[mask]) >= 0.0)