# -*- coding: utf-8 -*-
"""
Pitch Spelling using the ps13 algorithm.

References
----------


"""
import numpy as np
from collections import namedtuple

__all__ = ['estimate_spelling']

ChromamorpheticPitch = namedtuple('ChromamorpheticPitch', 'chromatic_pitch morphetic_pitch')

STEPS = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
UND_CHROMA = np.array([0, 2, 3, 5, 7, 8, 10], dtype=np.int)
ALTER = np.array(['n', '#', 'b'])


def estimate_spelling(note_array, method='ps13s1', *args, **kwargs):
    """Estimate pitch spelling using the ps13 algorithm [4]_, [5]_.

    Parameters
    ----------
    note_array : structured array
         Array with score information
    method : str (default 'ps13s1')
         Pitch spelling algorithm. More methods will be added.
    *args
        positional arguments for the algorithm specified in  `method`.
    **kwargs
        Keyword arguments for the algorithm specified in `method`.

    Returns
    -------
    spelling : structured array
        Array with pitch spellings. The fields are 'step', 'alter' and 
        'octave'

    References
    ----------
    .. [4] Meredith, D. (2006). "The ps13 Pitch Spelling Algorithm". Journal 
           of New Music Research, 35(2):121.
    .. [5] Meredith, D. (2019). "RecurSIA-RRT: Recursive translatable 
           point-set pattern discovery with removal of redundant translators". 
           12th International Workshop on Machine Learning and Music. Würzburg, 
           Germany.

    """
    if method == 'ps13s1':
        ps = ps13s1

    step, alter, octave = ps(note_array, *args, **kwargs)

    spelling = np.empty(len(step), dtype=[('step', 'U1'), ('alter', np.int), ('octave', np.int)])

    spelling['step'] = step
    spelling['alter'] = alter
    spelling['octave'] = octave

    return spelling


def ps13s1(note_array, K_pre=10, K_post=40):
    """
    ps13s1 Pitch Spelling Algorithm
    """
    pitch_sort_idx = note_array['pitch'].argsort()

    onset_sort_idx = np.argsort(note_array[pitch_sort_idx]['onset'], kind='mergesort')

    sort_idx = pitch_sort_idx[onset_sort_idx]

    re_idx = sort_idx.argsort()  # o_idx[sort_idx]

    sorted_ocp = np.column_stack(
        (note_array[sort_idx]['onset'],
         chromatic_pitch_from_midi(note_array[sort_idx]['pitch'])))

    n = len(sorted_ocp)
    # ChromaList
    chroma_array = compute_chroma_array(sorted_ocp=sorted_ocp)
    # ChromaVectorList
    chroma_vector_array = compute_chroma_vector_array(chroma_array=chroma_array,
                                                      K_pre=K_pre,
                                                      K_post=K_post)
    morph_array = compute_morph_array(chroma_array=chroma_array,
                                      chroma_vector_array=chroma_vector_array)

    morphetic_pitch = compute_morphetic_pitch(sorted_ocp, morph_array)

    step, alter, octave = p2pn(sorted_ocp[:, 1], morphetic_pitch.reshape(-1, ))
    # sort back pitch names
    step = step[re_idx]
    alter = alter[re_idx]
    octave = octave[re_idx]

    return step, alter, octave


def chromatic_pitch_from_midi(midi_pitch):
    return midi_pitch - 21


def chroma_from_chromatic_pitch(chromatic_pitch):
    return np.mod(chromatic_pitch, 12)


def pitch_class_from_chroma(chroma):
    return np.mod(chroma - 3, 12)


def compute_chroma_array(sorted_ocp):
    return chroma_from_chromatic_pitch(sorted_ocp[:, 1]).astype(np.int)


def compute_chroma_vector_array(chroma_array, K_pre, K_post):
    """
    Computes the chroma frequency distribution within the context surrounding
    each note.
    """
    n = len(chroma_array)
    chroma_vector = np.zeros(12, dtype=np.int)

    for i in range(np.minimum(n, K_post)):
        chroma_vector[chroma_array[i]] = 1 + chroma_vector[chroma_array[i]]

    chroma_vector_list = [chroma_vector.copy()]

    for i in range(1, n):
        if i + K_post <= n:
            chroma_vector[chroma_array[i + K_post - 1]] = 1 + chroma_vector[chroma_array[i + K_post - 1]]

        if i - K_pre > 0:
            chroma_vector[chroma_array[i - K_pre - 1]] = chroma_vector[chroma_array[i - K_pre - 1]] - 1

        chroma_vector_list.append(chroma_vector.copy())

    return np.array(chroma_vector_list)


def compute_morph_array(chroma_array, chroma_vector_array):

    n = len(chroma_array)
    # Line 1: Initialize morph array
    morph_array = np.empty(n, dtype=np.int)

    # Compute m0
    # Line 2
    init_morph = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6], dtype=np.int)
    # Line 3
    c0 = chroma_array[0]
    # Line 4
    m0 = init_morph[c0]

    # Line 5
    morph_int = np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6], dtype=np.int)

    # Lines 6-8
    tonic_morph_for_tonic_chroma = np.mod(m0 - morph_int[np.mod(c0 - np.arange(12), 12)], 7)

    # Line 10
    tonic_chroma_set_for_morph = [[] for i in range(7)]

    # Line 11
    morph_strength = np.zeros(7, dtype=np.int)

    # Line 12
    for j in range(n):
        # Lines 13-15 (skipped line 9, since we do not need to
        # initialize morph_for_tonic_chroma)
        morph_for_tonic_chroma = np.mod(morph_int[np.mod(chroma_array[j]
                                                         - np.arange(12), 12)] +
                                        tonic_morph_for_tonic_chroma, 7)
        # Lines 16-17
        tonic_chroma_set_for_morph = [[] for i in range(7)]

        # Line 18
        for m in range(7):
            # Line 19
            for ct in range(12):
                # Line 20
                if morph_for_tonic_chroma[ct] == m:
                    # Line 21
                    tonic_chroma_set_for_morph[m].append(ct)

        # Line 22
        for m in range(7):
            # Line 23
            morph_strength[m] = sum([chroma_vector_array[j, ct]
                                     for ct in tonic_chroma_set_for_morph[m]])

        # Line 24
        morph_array[j] = np.argmax(morph_strength)

    return morph_array


def compute_ocm_chord_list(sorted_ocp, chroma_array, morph_array):

    # Lines 1-3
    ocm_array = np.column_stack((sorted_ocp[:, 0], chroma_array, morph_array)).astype(np.int)

    # Alternative implementation of lines 4--9
    unique_onsets = np.unique(ocm_array[:, 0])
    unique_onset_idxs = [np.where(ocm_array[:, 0] == u) for u in unique_onsets]
    ocm_chord_list = [ocm_array[uix] for uix in unique_onset_idxs]

    return ocm_chord_list


def compute_morphetic_pitch(sorted_ocp, morph_array):
    """
    Compute morphetic pitch

    Parameters
    ----------
    sorted_ocp : array
       Sorted array of (onset in beats, chromatic pitch)
    morph_array : array
       Array of morphs

    Returns
    -------
    morphetic_pitch : array
        Morphetic pitch of the notes
    """
    n = len(sorted_ocp)
    chromatic_pitch = sorted_ocp[:, 1]
    morph = morph_array.reshape(-1, 1)

    morph_oct_1 = np.floor(chromatic_pitch / 12.0).astype(np.int)

    morph_octs = np.column_stack((morph_oct_1,
                                  morph_oct_1 + 1,
                                  morph_oct_1 - 1))

    chroma = np.mod(chromatic_pitch, 12)

    mps = morph_octs + (morph / 7)

    cp = (morph_oct_1 + (chroma / 12)).reshape(-1, 1)

    diffs = abs(cp - mps)

    best_morph_oct = morph_octs[np.arange(n), diffs.argmin(1)]

    morphetic_pitch = morph.reshape(-1, ) + 7 * best_morph_oct

    return morphetic_pitch


def p2pn(c_pitch, m_pitch):
    """
    Chromamorphetic pitch to pitch name

    Parameters
    ----------
    c_pitch : int or array
        Chromatic pitch.
    m_pitch : int or array
        Morphetic pitch.

    Returns
    -------
    step : str or array
        Note name (step)
    alter : int or array
        Alteration(s) of the notes. 1 is sharp, -1 is flat and 0 is natural
    octave : int or array
        Octave
    """
    morph = np.mod(m_pitch, 7)

    step = STEPS[morph]
    undisplaced_chroma = UND_CHROMA[morph]

    # displacement in paper
    alter = c_pitch - 12 * np.floor(m_pitch / 7.0) - undisplaced_chroma

    asa_octave = np.floor(m_pitch / 7)

    if isinstance(morph, (int, float)):
        if morph > 1:
            asa_octave += 1
    else:
        asa_octave[morph > 1] += 1

    return step, alter, asa_octave
