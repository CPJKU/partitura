#!/usr/bin/env python

import sys
import numpy as np
from operator import attrgetter
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, csc_matrix

from .scoreontology import Divisions

"""
Convert a musicxml part into sparse piano roll
"""

def notes_to_pianoroll_note_slices(notes, return_ioi=False):
    """
    Return a pianoroll where the n-th column is a slice of the pianoroll taken
    at the onset of the n-th note
    
    Parameters
    ----------
    notes: ndarray
       M x 3 ndarray representing M notes. The three columns specify
       onset, offset, and midi pitch of the notes
    
    Returns
    -------

    scipy.sparse.csr_matrix
        A sparse boolean matrix of size (128, len(`notes`)), representing the pianoroll

    """
    
    # round times to avoid problems with simultaneous on/off of triplets etc
    round_decimals = 4
    notes[:, 0] = np.round(notes[:, 0], decimals=round_decimals)
    notes[:, 1] = np.round(notes[:, 1], decimals=round_decimals)

    ongoing = defaultdict(lambda *x: defaultdict(list))
    for n in notes:
        p = int(n[2])
        ongoing[p]['onset'].append(n[0])
        ongoing[p]['offset'].append(n[1])
    pitches = sorted(ongoing.keys())
    hslices = {}
    pr = lil_matrix((128, len(notes)), dtype=np.bool)
    for p in pitches:
        ons = np.array(ongoing[p]['onset'])
        offs = np.array(ongoing[p]['offset'])
        onoffs = np.column_stack((np.r_[ons, offs],
                                  np.r_[np.ones_like(ons),
                                        np.zeros_like(offs)]))
        onoffs = onoffs[np.lexsort((onoffs[:, 1], onoffs[:, 0]))]

        # remove simultaneous on/off of pitch p
        to_remove = np.where(np.diff(onoffs[:, 0]) <= 0)[0]
        onoffs[to_remove, 1] = 1
        #to_remove = set.union(set(to_remove), set(to_remove+1))
        onoffs = np.delete(onoffs, np.array(sorted(to_remove)), axis=0)

        idx = np.where(interp1d(onoffs[:, 0], onoffs[:, 1], kind='zero',
                                bounds_error=False, fill_value=0)(notes[:, 0]) > 0)[0]
        pr[p, idx] = True
    # z = np.where(np.sum(pr.toarray(), 0) == 0)[0]
    # k = 3
    # print(z)
    # for i in z:
    #     print(notes[i-k:i+k,:])
    if return_ioi:
        ioi_dict = dict((np.int(k), v) for k, v in
                        np.column_stack((np.unique(notes[:, 0])*10**round_decimals,
                                         np.diff(np.r_[np.unique(notes[:, 0]), notes[-1, 1]]))))
        return pr.tocsr(), np.array([ioi_dict[np.int(k*10**round_decimals)] for k in notes[:, 0]])
    else:
        return pr.tocsr()


def scorepart_to_notes(part, morphetic_pitch=False):
    """Convert a score part to a list of notes. When the score part contains N
    notes, this function returns an N x 3 array, containing onset, offset, and
    midi pitch information.

    :param part: Score part
    :returns: Note information
    :rtype: ndarray

    """

    bm = part.beat_map

    if morphetic_pitch:
        pitch_attr = attrgetter('morphetic_pitch')
    else:
        pitch_attr = attrgetter('midi_pitch')
    notes = np.array([(bm(n.start.t), bm(n.end.t), pitch_attr(n))
                      for n in part.notes], np.float)

    ids = np.array([n.id for n in part.notes])

    return notes, ids


def notes_to_pianoroll(notes, onset_only=True,
                       pitch_margin=-1, beat_margin=0,
                       beat_div=8):
    """
    Convert note information to a pianoroll.
    
    Parameters
    ----------
    notes: ndarray
       M x 3 ndarray representing M notes. The three columns specify onset,
       offset, and midi pitch of the notes

    onset_only: type, optional
       if True, code only the onsets of the notes, otherwise code onset and
       duration

    pitch_margin: type, optional
       if `pitch_margin` > -1, the resulting array will have `pitch_margin`
       empty rows above and below the highest and lowest pitches, respectively;
       if `pitch_margin` == -1, the resulting pianoroll will have span the fixed
       pitch range between (and including) 1 and 127.

    beat_margin: type, optional
       the resulting array will have `beat_margin` * `beat_div` empty columns
       before and after the piano roll

    beat_div: type, optional
       how many sub-divisions for each beat
    
    Returns
    -------

    scipy.sparse.csr_matrix
        A sparse boolean matrix of size representing the pianoroll; The first
        dimension is pitch, the second is time; The sizes of the dimensions vary
        with the parameters `pitch_margin`, `beat_margin`, and `beat_div`
    """

    m, _ = _notes_to_pianoroll(notes, onset_only,
                               pitch_margin, beat_margin, beat_div)
    return m


def notes_to_notecentered_pianoroll(notes, onset_only=True,
                                    neighbour_pitches=10, neighbour_beats=2,
                                    beat_div=8):
    """Convert a matrix with note information (3 columns: onset, offset, midipitch)
    into a sparse note-centered piano roll representation, and return the result
    as a sparse matrix

    :param notes: M x 3 array representing M notes

    :param onset_only: if True, code only the onsets of the notes, otherwise
       code onset and duration

    :param lowest_pitch: the lowest midi pitch to consider

    :param highest_pitch: the highest midi pitch to consider

    :param beat_div: how many sub-divisions for each beat

    :param neighbour_beats: how many beats before and after the central note to
       code

    :returns: a tuple (A, pitch_range), where A is a M x N sparse matrix where
       each row corresponds to a vectorized representation of a note context. N
       = (2 * (`highest_pitch` - `lowest_pitch`)) * `beat_div` * (2 *
       `neighbour_beats`), and pitch_range is the size of the first dimension
       (pitch) of the note contexts. So in order to reconstruct the 2D note
       context of the i-th note from A, do: A[i,:].reshape((pitch_range, -1))

    """

    m, idx = _notes_to_pianoroll(notes, onset_only,
                                 neighbour_pitches, neighbour_beats, beat_div)

    return _note_center(neighbour_pitches, neighbour_beats, beat_div, idx, m)


def _notes_to_pianoroll(notes, onset_only,
                        neighbour_pitches, neighbour_beats, beat_div):

    # columns:
    ONSET = 0
    OFFSET = 1
    PITCH = 2

    if neighbour_pitches > -1:
        highest_pitch = np.max(notes[:, PITCH])
        lowest_pitch = np.min(notes[:, PITCH])
    else:
        lowest_pitch = 0
        highest_pitch = 127

    pitch_span = highest_pitch - lowest_pitch + 1

    # sorted idx
    idx = np.argsort(notes[:, ONSET])
    # sort notes
    notes = notes[idx, :]
    min_time = notes[0, ONSET]
    max_time = np.max(notes[:, OFFSET])

    # shift times to start at 0
    notes[:, ONSET] -= min_time - neighbour_beats
    notes[:, OFFSET] -= min_time - neighbour_beats
    if neighbour_pitches > -1:
        notes[:, PITCH] -= lowest_pitch
        notes[:, PITCH] += neighbour_pitches

    # size of the feature matrix
    if neighbour_pitches > -1:
        M = int(pitch_span + 2 * neighbour_pitches)
    else:
        M = int(pitch_span)
    N = int(np.ceil(beat_div * (2 * neighbour_beats + max_time - min_time)))

    nnotes = notes.copy()
    nnotes[:, (ONSET, OFFSET)] = np.round(beat_div * notes[:, (ONSET, OFFSET)])
    nnotes = nnotes.astype(np.int)
    idx = nnotes[:, (PITCH, ONSET)]

    if onset_only:
        idx_fill = idx
    else:
        nnotes[:, OFFSET] = np.maximum(nnotes[:, ONSET] + 1,
                                       nnotes[:, OFFSET] - 1)
        idx_fill = np.vstack([np.column_stack((np.zeros(off - on) + pitch,
                                               np.arange(on, off)))
                              for on, off, pitch in nnotes
                              if off < N])
        # off >= N does not normally happen, typically only when grace notes at
        # the end are expanded
    m = csc_matrix((np.ones(idx_fill.shape[0]),
                    (idx_fill[:, 0], idx_fill[:, 1])),
                   shape=(M, N), dtype=np.bool)

    return m, idx


def _note_center(neighbour_pitches, neighbour_beats, beat_div, idx, m):

    neighbour_timesteps = beat_div * neighbour_beats
    pt_nbh = np.array((neighbour_pitches, neighbour_timesteps))

    vslice = None
    t_min_prev = None
    K = 2 * neighbour_timesteps

    nn = []
    mm = []
    for i, n in enumerate(idx):
        p_min, t_min = n - pt_nbh
        p_max, t_max = n + pt_nbh

        # subsequent notes with the same onset reuse the vslice
        if t_min_prev != t_min:
            vslice = m[:, t_min:t_max]

        t_min_prev = t_min

        ii, jj = vslice[p_min:p_max + 1, :].nonzero()

        mm.append(ii * K + jj)
        nn.append(np.zeros(ii.shape[0]) + i)

    mm = np.concatenate(mm)
    nn = np.concatenate(nn)

    A = csr_matrix((np.ones(nn.shape[0]),
                    (nn, mm)),
                   shape=(
                       idx.shape[0], (2 * neighbour_pitches + 1) * (2 * neighbour_timesteps)),
                   dtype=np.bool)
    return A  # , 2 * neighbour_pitches + 1


if __name__ == '__main__':
    pass
