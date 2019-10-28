#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Voice Separation using Chew and Wu's algorithm.

"""
import numpy as np

from collections import defaultdict
from numpy import ma
from statistics import mode

from partitura.utils import add_field

__all__ = ['estimate_voices']

# Maximal cost of a jump (in Chew  and Wu (2004) is 2 ** 31)
MAX_COST = 1000


def rename_voices(voices):
    # rename voices so that the first occurring voice has number 1, the second
    # occurring voice has number 2, etc.
    vmap = {}
    return np.fromiter((vmap.setdefault(v, len(vmap) + 1) for v in voices),
                       dtype=voices.dtype)


def prepare_notearray(notearray):
    # * check whether notearray is a structured array
    # * check whether it has pitch/onset/duration fields
    # * return a copy of pitch/onset/duration fields with added id field
    if notearray.dtype.fields is None:
        raise ValueError('`notearray` must be a structured numpy array')

    req_fields = ('pitch', 'onset', 'duration')
    for field in req_fields:
        if field not in notearray.dtype.names:
            raise ValueError('Input array does not contain required field {0}'.format(field))

    dtypes = dict(notearray.dtype.descr)
    new_dtype = [(n, dtypes[n]) for n in req_fields] + [('id', 'i4')]

    return np.fromiter(zip(notearray['pitch'],
                           notearray['onset'],
                           notearray['duration'],
                           np.arange(len(notearray))),
                       dtype=new_dtype)


def argmax_pitch(idx, pitches):
    return idx[np.argmax(pitches[idx])]


def estimate_voices(notearray, monophonic_voices=False):
    """Voice estimation using the voice separation algorithm proposed 
    in [1]_.

    Parameters
    ----------
    notearray : numpy structured array
        Structured array containing score information.
        Required fields are `pitch` (MIDI pitch),
        `onset` (starting time of the notes) and
        `duration` (duration of the notes). Additionally,
        It might be useful to have an `id` field containing
        the ID's of the notes. If this field is not contained
        in the array, ID's will be created for the notes.
    monophonic_voices : bool
        If True voices are guaranteed to be monophonic. Otherwise
        notes with the same onset and duration are treated as a chord
        and assigned to the same voice. Defaults to False.

    Returns
    -------
    voice : numpy array
        Voice for each note in the notearray. (The voices start with 1, as
        is the MusicXML convention).

    References
    ----------
    .. [1] Chew, E. and Wu, Xiaodan (2004) "Separating Voices in
           Polyphonic Music: A Contig Mapping Approach". In Uffe Kock, 
           editor, "Computer Music Modeling and Retrieval". Springer 
           Berlin Heidelberg.

    TODO
    ----
    * Handle grace notes correctly. The current version simply
      deletes all grace notes.

    """

    input_array = prepare_notearray(notearray)

    # Remove grace notes
    # grace_note_idxs = np.where(input_array['duration'] == 0)[0]

    # grace_by_key = defaultdict(list)

    # for (pitch, onset, dur, i) in input_array[grace_note_idxs]:
    #     grace_by_key[i].append(

    if monophonic_voices:

        # identity mapping
        idx_equivs = dict((n, n) for n in input_array['id'])

    else:

        note_by_key = defaultdict(list)

        for (pitch, onset, dur, i) in input_array:
            note_by_key[(onset, dur)].append(i)

        # dict that maps first chord note index to the list of all note indices
        # of the same chord
        idx_equivs = dict((argmax_pitch(np.array(idx), input_array['pitch']), np.array(idx))
                          for idx in note_by_key.values())

        # keep the first note of each chord, the rest of the chord notes will be
        # assigned the same voice as the first chord note
        input_array = input_array[sorted(idx_equivs.keys())]

    # Perform voice separation
    v_notearray = VoSA(input_array).note_array

    # map the voices to the original notes
    voices = np.empty(len(notearray), dtype=np.int)
    for idx, voice in zip(v_notearray['id'], v_notearray['voice']):
        voices[idx_equivs[idx]] = voice

    # rename voices so that the first occurring voice has number 1, the second
    # occurring voice has number 2, etc.
    rvoices = rename_voices(voices)

    return rvoices


def pairwise_cost(prev, nxt):
    """
    Compute pairwise cost between two contigs

    Parameters
    ----------
    prev : Contig, VoiceManager or iterable
        VSNotes in the left side of the connection.

    nxt : Contig, VoiceManager or iterable
        VSNotes in the right side of the connection.

    Returns
    -------
    cost : np.ndarray
        Cost of connecting each note in the last onset of the
        previous of the contig to each note in the first onset
        of the next contig. (index [i, j] represents the cost
        from connecting i in `prev` to  j in `nxt`.
    """
    # Get previous notes according to the type of `prev`
    if isinstance(prev, Contig):
        n_prev_voices = len(prev.last)
        prev_notes = prev.last
    elif isinstance(prev, VoiceManager):
        n_prev_voices = prev.num_voices
        prev_notes = [v.last for v in prev]
    elif isinstance(prev, (list, tuple, np.ndarray)):
        n_prev_voices = len(prev)
        prev_notes = prev

    # Get the next notes according to the type of `nxt`
    if isinstance(nxt, Contig):
        n_nxt_voices = len(nxt.first)
        next_notes = nxt.first
    elif isinstance(nxt, VoiceManager):
        n_nxt_voices = nxt.num_voices
        next_notes = [v.first for v in nxt]
    elif isinstance(nxt, (list, tuple, np.ndarray)):
        n_nxt_voices = len(nxt)
        next_notes = nxt

    # Initialize cost array
    cost = np.zeros((n_prev_voices, n_nxt_voices))

    # Compute cost
    for i, c_note in enumerate(prev_notes):
        for j, n_note in enumerate(next_notes):
            if c_note == n_note:
                cost[i, j] = - MAX_COST
            elif c_note.skip_contig != 0 or n_note.skip_contig != 0:
                cost[i, j] = MAX_COST
            else:
                cost[i, j] = abs(c_note.pitch - n_note.pitch)

    return cost


def est_best_connections(cost, mode='prev'):
    """
    Get the connections with minimal cost

    Parameters
    ----------
    cost : np.ndarray
        Cost of connecting two contigs. See `pairwise_cost`.
    mode : 'prev' or 'next'
        Whether the connection is from the previous to the next
        or from the next to the previous.

    Returns
    -------
    best_assignment : np.ndarray
        2D array where the first column are the streams in the
        first contig (previous if mode is 'prev' or next if the
        mode is 'next') and the second column are the corresponding
        stream in the second contig (next if mode is 'prev' and previous
        if mode is 'next').
    unassigned_streams : list
        Unassigned streams in the previous contig.
    """

    # number of streams in the first and second contigs
    n_streams_p, n_streams_n = cost.shape

    # determine sizes according to the mode
    if mode == 'prev':
        con_cost = cost
        n_streams = n_streams_p
        n_assignments = n_streams_n
    elif mode == 'next':
        con_cost = cost.T
        n_streams = n_streams_n
        n_assignments = n_streams_p

    # initialize mask for the cost
    mask = np.zeros_like(con_cost)
    mcost = ma.masked_array(con_cost, mask=mask)

    # Initialize list of best assignments
    best_assignment = []

    # while there are fewer than n_assignments
    while len(best_assignment) < n_assignments:

        # Get the remaining minimal cost
        next_best = mcost.min(1).argmin()
        next_assig = mcost.argmin(1)[next_best]

        # append minimal assignment to the list
        best_assignment.append((next_best, next_assig))

        # Mask this assignment so that it cannot be considered
        # in the next step of the loop
        mask[:, next_assig] = 1
        mask[next_best, :] = 1
        mcost.mask = mask

    best_assignment = np.array(best_assignment).astype(np.int)

    # Get unassigned streams
    unassigned_streams = list(set(range(n_streams)).difference(best_assignment[:, 0]))

    return best_assignment, unassigned_streams


def sort_by_pitch(sounding_notes):
    """
    Sort a list of notes by pitch

    Parameters
    ----------
    sounding_notes : list
        List of `VSNote` instances

    Returns
    -------
    list
        List of sounding notes sorted by pitch
    """
    return sorted(sounding_notes, key=lambda x: x.pitch)


def sort_by_onset(sounding_notes):
    """
    Sort a list of notes by pitch

    Parameters
    ----------
    sounding_notes : list
        List of `VSNote` instances

    Returns
    -------
    list
        List of sounding notes sorted by onset
    """
    return sorted(sounding_notes, key=lambda x: x.onset)


class VSNote(object):
    """Base class to hold a Note for a voice separation algorithm

    Parameters
    ----------
    pitch : int or float
        MIDI pitch of the note
    onset : float
        Score onset time in beats
    duration : float
        Notated duration in beats
    note_id : str or int
        ID of the note
    velocity : int or `None` (optional)
        MIDI Velocity of the note. Default is `None`

    Attributes
    ----------
    id : int or str
        Identifier of the note
    pitch : int or float
        MIDI pitch of the note
    onset : float
        Onset in beats
    duration : float
        Duration in beats
    offset : float
        Offset in beats
    skip_contig : int or bool
        If the note belongs to a stream that was not connected to
        its immediate neighbor
    is_grace : bool
        Whether the note is a grace note
    voice : int
        Voice of the note. Setting this attribute also sets the voice
        of associated grace notes (experimental...)
    velocity : int or `None`
        MIDI velocity of the note
    """

    def __init__(self, pitch, onset, duration, note_id, velocity=None,
                 voice=None):

        # ID of the note
        self.id = note_id
        self.pitch = pitch
        self.onset = onset
        self.duration = duration
        self.offset = onset + duration

        self.skip_contig = 0

        self.is_grace = self.duration == 0
        self._grace = []
        self._voice = voice
        self.velocity = None

    @property
    def voice(self):
        return self._voice

    @voice.setter
    def voice(self, voice):
        self._voice = voice
        for n in self._grace:
            n.voice = voice

    def is_sounding(self, tp):
        return tp >= self.onset and tp < self.offset

    @property
    def grace(self):
        return self._grace

    @grace.setter
    def grace(self, grace):
        self._grace.append(grace)
        for n in self._grace:
            n.voice = self.voice

    def __str__(self):
        return 'VSNote {id}: pitch {pi}, onset {on}, duration {dur}, voice {voice}'.format(
            id=self.id,
            pi=self.pitch,
            on=self.onset,
            dur=self.duration,
            voice=self.voice if self.voice is not None else 'None')


class VSChord(object):
    """
    Base class to hold chords
    """

    def __init__(self, notes, rep_note='highest'):

        if any([n.onset != notes[0].onset for n in notes]):
            raise ValueError('All notes in the chord must have the same onset')
        if any([n.offset != notes[0].offset for n in notes]):
            raise ValueError('All notes in the chord must have the same offset')
        self.notes = notes

        self.pitches = np.array([n.pitch for n in self.notes])
        self.onset = self.notes[0].onset
        self.offset = self.notes[0].offset
        self.duration = self.notes[0].duration

        self.velocity = [n.velocity for n in self.notes]

        self.rep_note = rep_note

    @property
    def pitch(self):
        if self.rep_note == 'highest':
            return self.pitches.max()

        elif self.rep_note == 'lowest':
            return self.pitches.min()

        elif isinstance(self.rep_note, int):
            return self.pitches[self.rep_note]

    @property
    def voice(self):
        # The entire chord is assigned to the same voice
        return self.notes[0].voice

    @voice.setter
    def voice(self, voice):
        for n in self.notes:
            n.voice = voice

    def is_sounding(self, tp):
        return tp >= self.onset and tp < self.offset


class Voice(object):
    """
    Class to hold a voice as a list of NoteStream
    """

    def __init__(self, stream_or_streams, voice=None):

        if isinstance(stream_or_streams, list):
            self.streams = stream_or_streams
        elif isinstance(stream_or_streams, NoteStream):
            self.streams = [stream_or_streams]

        if len(self.streams) > 0:
            self._setup_voice()

        else:
            self.notes = []

        self._voice = voice

    def _setup_voice(self):

        # sort stream by onset
        self.streams.sort(key=lambda x: x.onset)

        # array notes in the stream
        self.notes = []
        for stream in self.streams:
            self.notes += list(stream.notes)

        self.notes = np.array(sort_by_onset(self.notes))

        self.streams[0].prev_stream = None
        self.streams[-1].next_stream = None
        for i, stream in enumerate(self.streams[1:]):
            stream.prev_stream = self.streams[i]
            self.streams[i].next_stream = stream

        self.voice = self._voice

    @property
    def onset(self):
        return self.streams[0].onset

    @property
    def offset(self):
        return self.streams[-1].offset

    @property
    def duration(self):
        return self.offset - self.onset

    def append(self, stream):
        stream.voice = self.voice
        self.streams.append(stream)

        self._setup_voice()

    @property
    def voice(self):
        return self._voice

    @voice.setter
    def voice(self, voice):
        self._voice = voice
        for n in self.notes:
            n.voice = voice

    @property
    def first(self):
        # First note in the voice
        return self.streams[0].first

    @property
    def last(self):
        # Last note in the voice
        return self.streams[-1].last


class VoiceManager(object):
    """
    Manage the progress of several voices
    """

    def __init__(self, num_voices):

        self.num_voices = num_voices

        self.voices = [Voice([], voice) for voice in range(self.num_voices)]

    def __getitem__(self, index):
        return self.voices[index]

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx == len(self.voices):
            raise StopIteration
        res = self[self.iter_idx]
        self.iter_idx += 1
        return res

    def __len__(self):
        return len(self.voices)


class VSBaseScore(object):
    """
    Base class for holding score-like objects for voice separation
    """

    def __init__(self, notes):

        # Set list of notes
        self.notes = notes

        if len(self.notes) > 0:
            self._setup_score()

    def _setup_score(self):
        if isinstance(self.notes, list):
            self.notes = np.array(list(set(self.notes)))
        elif isinstance(self.notes, np.ndarray):
            self.notes = np.array(list(set(self.notes)))

        # sort notes by onset
        self.notes = self.notes[np.argsort([n.onset for n in self.notes])]
        # Get onsets of the notes
        self.note_onsets = np.array([n.onset for n in self.notes])
        # Get offsets of the notes
        self.note_offsets = np.array([n.offset for n in self.notes])

        # Get unique onsets
        self.unique_onsets = np.unique(self.note_onsets)
        self.unique_onsets.sort()

        # Get duration of the notes
        self.note_durations = self.note_offsets - self.note_onsets

        # Get all timepoints in the score
        self.unique_timepoints = np.unique(np.hstack((self.note_onsets, self.note_offsets)))
        # Sort them in ascending order
        self.unique_timepoints.sort()

        # shortcut
        self.utp = self.unique_timepoints

        # Initialize dictionary of sounding notes at each unique time point
        self._sounding_notes = dict()

        for tp in self.unique_timepoints:
            # boolean array of the notes ending after the time point
            # (and therefore sounding at this timepoint)
            ending_after_tp = self.note_offsets > tp

            # boolean array of the notes starting before or at this time point
            starting_before_tp = self.note_onsets <= tp

            # boolean array of the notes starting at this timepoint
            sounding_idxs = np.logical_and(starting_before_tp, ending_after_tp)

            # Set notes in dictionary
            self._sounding_notes[tp] = sort_by_pitch(list(self.notes[sounding_idxs]))

    def __getitem__(self, index):
        """Get element in the score by index of the time points.
        """
        return sort_by_pitch(self._sounding_notes[self.unique_timepoints[index]])

    def __setitem__(self, index, notes):
        if isinstance(notes, list):
            self._sounding_notes[self.unique_timepoints[index]] = notes.copy()
        elif isinstance(notes, VSNote):
            self._sounding_notes[self.unique_timepoints[index]] = [notes.copy()]

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx == len(self.unique_timepoints):
            raise StopIteration
        res = self[self.iter_idx]
        self.iter_idx += 1
        return res

    def sounding_notes(self, tp):
        """
        Get all sounding notes at a specific timepoint

        Parameters
        ----------
        tp : float
            Timepoint at which we want to get all sounding notes.

        Returns:
        notes : list
            List of sounding notes at timepoint `tp` sorted by pitch.
        """
        s_ix = np.max(np.where(self.unique_timepoints <= tp)[0])

        return sort_by_pitch(self._sounding_notes[self.unique_timepoints[s_ix]])

    def num_sound_notes(self, tp):
        return len(self.sounding_notes(tp))

    def __len__(self):
        return len(self.unique_timepoints)


class NoteStream(VSBaseScore):
    def __init__(self, notes=[], voice='auto', prev_stream=None, next_stream=None):

        super(NoteStream, self).__init__(notes)
        self._voice = voice
        self.prev_stream = prev_stream
        self.next_stream = next_stream

        if len(self.notes) > 0:
            if self._voice == 'auto':
                self.infer_voice()

        self.onset = self.note_onsets.min()
        self.offset = self.note_offsets.max()

    def append(self, note):
        # only append if the note is not already in the stream
        # to avoid unnecessay duplications
        if note not in self.notes:
            if isinstance(self.notes, list):
                if isinstance(note, VSNote):
                    self.notes.append(note)
                elif isinstance(note, list):
                    self.notes += note
                elif isinstance(note, np.ndarray):
                    self.notes = np.append(self.notes, note)
            elif isinstance(self.notes, np.ndarray):
                self.notes = np.append(self.notes, note)
            # update note stream
            self._setup_score()

    def infer_voice(self):
        voices = [n.voice for n in self.notes]
        self.voice = mode(voices)

    @property
    def voice(self):
        return self._voice

    @voice.setter
    def voice(self, voice):
        self._voice = voice

        if len(self.notes) > 0:
            for note in self.notes:
                note.voice = self._voice

    @property
    def first(self):
        # Check method...
        # What would happen in the case of several notes?
        # Perhaps make chords for stream?
        return self.notes[self.note_onsets.argmin()]

    @property
    def last(self):
        return self.notes[self.note_onsets.argmax()]


class Contig(VSBaseScore):
    def __init__(self, notes, is_maxcontig=False):

        super(Contig, self).__init__(notes)
        self._is_maxcontig = is_maxcontig
        # Onset time of the contig
        self.n_voices = np.array([self.num_sound_notes(tp) for tp in self.utp])

        self.onset = self.utp[np.where(self.n_voices == self.n_voices.max())].min()
        self.n_voices = self.n_voices.max()
        # offset of the contig is the minimum offset of all
        # notes in the last onset of the contig
        self.offset = min([n.offset for n in self._sounding_notes[self.note_onsets.max()]])
        self.duration = self.offset - self.offset

        # initialize streams as a list for each voice in the contig
        streams = [[] for i in range(self.n_voices)]

        for on in self.unique_onsets[np.where(self.unique_onsets >= self.onset)[0]]:
            for i, note in enumerate(self.sounding_notes(on)):

                if note not in streams[i]:
                    streams[i].append(note)

        self.streams = [NoteStream(stream) for stream in streams]

        # set voices for if the contig is maximal
        self._set_voices_for_maxcontig()

    @property
    def is_maxcontig(self):
        return self._is_maxcontig

    @is_maxcontig.setter
    def is_maxcontig(self, is_maxcontig):
        self._is_maxcontig = bool(is_maxcontig)
        self._set_voices_for_maxcontig()

    def _set_voices_for_maxcontig(self):
        # sets the voice for maximal contigs
        if self._is_maxcontig:
            for vn, stream in enumerate(self.streams):
                stream.voice = vn

    @property
    def first(self):
        "first onset in the contig"
        sounding_notes = [n for n in self.sounding_notes(self.onset)]
        return sort_by_pitch(sounding_notes)

    @property
    def last(self):
        sounding_notes = [n for n in self.sounding_notes(self.unique_onsets.max())]
        return sort_by_pitch(sounding_notes)

    @property
    def has_voice_info(self):
        return all([stream.voice is not None for stream in self.streams])


class VoSA(VSBaseScore):
    """Class to represent a score for voice separation

    TODO:
    * rename this class or simplify to avoid overlap in naming
      conventions with the main package
    * better handle grace notes
    """

    def __init__(self, score, delete_gracenotes=False):

        # Score
        self.score = score

        if delete_gracenotes:
            # TODO: Handle grace notes correctly
            self.score = self.score[score['duration'] != 0]
        else:
            grace_note_idxs = np.where(score['duration'] == 0)[0]
            unique_onsets = np.unique(self.score['onset'])
            unique_onset_idxs = [np.where(self.score['onset'] == u)[0]
                                 for u in unique_onsets]
            main_notes_idxs = []
            grace_notes = []
            for g_i in grace_note_idxs:
                grace_note = self.score[g_i]
                candidate_note_idxs = np.where(self.score['onset'] == grace_note['onset'])[0]
                candidate_note_idxs = candidate_note_idxs[candidate_note_idxs != g_i]

                if len(candidate_note_idxs) == 0:
                    next_onset_idx = int(np.where(unique_onsets == grace_note['onset'])[0] + 1)
                    candidate_note_idxs = unique_onset_idxs[next_onset_idx]

                candidate_notes = self.score[candidate_note_idxs]
                main_notes_idxs.append(candidate_note_idxs[
                    np.argmin(abs(candidate_notes['pitch'] -
                                  grace_note['pitch']))])

        self.notes = []

        for n in self.score:

            note = VSNote(pitch=n['pitch'],
                          onset=n['onset'],
                          duration=n['duration'],
                          note_id=n['id'])

            self.notes.append(note)

        # import pdb
        # pdb.set_trace()
        if not delete_gracenotes:
            for g_i, m_i in zip(grace_note_idxs, main_notes_idxs):
                self.notes[g_i].is_grace = True
                self.notes[m_i].grace = self.notes[g_i]

        self.notes = np.array(sort_by_onset(self.notes))

        super(VoSA, self).__init__(self.notes)

        self.contigs = None

        self.make_contigs()
        self.estimate_voices()

    def _build_streams(self):
        self.voices = []
        for vn in range(self.num_voices):
            notes_per_voice = [note for note in self.notes if note.voice == vn]
            self.voices.append(NoteStream(notes_per_voice, voice=vn))

    @property
    def note_array(self):
        """
        TODO:
        Check that all notes have the same type of id
        """
        out_array = []

        for n in self.notes:
            out_note = (n.pitch, n.onset, n.duration,
                        n.voice if n.voice is not None else -1, n.id)
            out_array.append(out_note)

        return np.array(out_array, dtype=[('pitch', 'i4'),
                                          ('onset', 'f4'),
                                          ('duration', 'f4'),
                                          ('voice', 'i4'),
                                          ('id', type(self.notes[0].id))]
                        )

    def make_contigs(self):

        # number of voices at each time point in the score
        n_voices = np.array([len(sn) for sn in self])

        self.num_voices = np.max(n_voices)

        # if len(n_voices) > 5:
        #     if n_voices[:-3].max() < n_voices[-3:].max():
        #         self.num_voices = n_voices[:-3].max()
        #     else:
        #         self.num_voices = np.max(n_voices)

        # else:
        #     self.num_voices = np.max(n_voices)

        # change in number of voices
        # it includes the beginning (there were no notes before the begining)
        # and the ending (there are no active voices at the end of the piece)
        n_voice_changes = np.r_[len(self[0]), np.diff(n_voices)]

        # initialize segment boundaries with changes in number of voices
        segment_boundaries = n_voice_changes != 0

        # Look for voice status changes
        for i, sn in enumerate(self):

            # an array of booleans that indicate whether each currently sounding note
            # was sounding in the previous time point (segment)
            note_sounding_in_prev_segment = np.array([n in self[i - 1] for n in sn])

            # Update the segment boundary if:
            # * there are sounding notes in the previous segment; and
            # * there is a change in number of voices in the current segment
            if any(note_sounding_in_prev_segment) and n_voice_changes[i] != 0:

                # indices of the sounding notes belonging to the previous time point
                prev_sounding_note_idxs = np.where(note_sounding_in_prev_segment)[0]

                # Update segment boundaries
                for psnix in prev_sounding_note_idxs:

                    # get sounding note
                    prev_sounding_note = sn[psnix]
                    # Get index of the timepoint of the onset of the prev_sounding_note
                    timepoint_idx = np.where(self.utp == prev_sounding_note.onset)[0]

                    # update segment boundaries
                    segment_boundaries[timepoint_idx] = True

        # initialize dictionary for contigs
        self.contig_dict = dict()
        # List for the number of voices per contig
        self._voices_per_contig = []
        # Initial onset of the contig
        self._contigs_init_onsets = []

        # iterate over timepoints, sounding notes and segment boundaries
        for tp, sn, sb, nv in zip(self.utp, self, segment_boundaries, n_voices):

            # If there is a segment boundary and there are sounding notes
            # (i.e. do not make empty contigs)
            if sb and len(sn) > 0:
                # initialize the contig
                self.contig_dict[tp] = sn
                # keep the timepoint
                last_tp = tp
                self._voices_per_contig.append(nv)
                self._contigs_init_onsets.append(tp)

            # in case that at the current timepoint there is no boundary
            else:
                # for each sounding note just append the new notes (avoid duplicate
                # notes). Please notice that the sounding notes are
                # duplicated if they cross segment boundaries
                for n in sn:
                    if n not in self.contig_dict[last_tp]:
                        self.contig_dict[last_tp].append(n)

        self._contigs_init_onsets.sort()

        self.contigs = []
        for tp in self._contigs_init_onsets:
            # Create `Contig` instances for each list of notes in a contig
            self.contig_dict[tp] = Contig(self.contig_dict[tp])
            self.contigs.append(self.contig_dict[tp])

    def estimate_voices(self):
        """
        Estimate voices using global minimum connections
        """
        # indices of the maximal contigs
        maximal_contigs_idxs = np.where(self._voices_per_contig == self.num_voices)[0]

        # indices of the non maximal contigs
        non_maximal_contigs_idx = np.where(self._voices_per_contig != self.num_voices)[0]

        # initialize maximal contigs and voice managers
        voice_managers_dict = dict()
        for mci in maximal_contigs_idxs:
            voice_managers_dict[mci] = VoiceManager(self.num_voices)
            # Initialize the maximal contigs
            self.contigs[mci].is_maxcontig = True

            # append the maximal contigs to the voice managers
            for s_i, stream in enumerate(self.contigs[mci].streams):
                voice_managers_dict[mci].voices[s_i].append(stream)

        # index of the neighbor contig (start with immediate contigs)
        nix = 1
        keep_loop = True
        # Initialize list for unassigned connections (forward and backward)
        f_unassigned = []
        b_unassigned = []

        # The loop iterates until all notes have been assigned a voice,
        # or there is no more score left
        while keep_loop:
            # Cristalization process around the maximal contigs
            for mci in maximal_contigs_idxs:

                # Get voice manager corresponding to the current
                # maximal contig
                vm = voice_managers_dict[mci]
                try:
                    # forward contig
                    f_contig = self.contigs[mci + (nix - 1)]
                except IndexError:
                    # if there are no more contigs (i.e. the end of the piece)
                    f_contig = None

                try:
                    # backward contig
                    b_contig = self.contigs[mci - (nix - 1)]
                except IndexError:
                    # if there are no more contigs (i.e. the beginning of the piece)
                    b_contig = None

                try:
                    # next neighbor contig
                    next_contig = self.contigs[mci + nix]
                except IndexError:
                    next_contig = None
                try:
                    # previous neighbor contig
                    prev_contig = self.contigs[mci - nix]
                except IndexError:
                    prev_contig = None

                # If we have not reached the end of the piece
                if f_contig is not None:
                    # If there is still a next contig
                    if next_contig is not None:
                        # If the next neighbor contig has not yet
                        # been assigned (assigne voice wrt the closest
                        # maximal contig)
                        if not next_contig.has_voice_info:

                            # flag voices without a connection in
                            # the previous step in the loop
                            for es in f_unassigned:
                                vm[es].last.skip_contig += 1

                            # Compute connection cost
                            cost = pairwise_cost(vm, next_contig)
                            # Estimate best connections (global minimum policy)
                            best_connections, f_unassigned = est_best_connections(cost, mode='prev')
                            # for s in vm:
                            #     s.last.skip_contig = 0

                            # Extend voices with corresponding stream
                            for es, ns in best_connections:
                                vm[es].append(next_contig.streams[ns])

                            # for es in f_unassigned:
                            #     vm[es].last.skip_contig += 1

                # If we have not reached the beginning of the piece
                if b_contig is not None:
                    # If there is still a previous contig
                    if prev_contig is not None:
                        # If the voices in the previous neighbor contig have
                        # not yet been assigned (assigne voces wrt to the
                        # closest maximal contig)
                        if not prev_contig.has_voice_info:
                            # flag voices without a connection in the
                            # previous step in the loop
                            for es in b_unassigned:
                                vm[es].first.skip_contig += 1

                            # Compute connection cost
                            cost = pairwise_cost(prev_contig, vm)
                            # Estimate best connections
                            best_connections, b_unassigned = est_best_connections(cost, mode='next')
                            # for s in vm:
                            #     s.first.skip_contig = 0

                            # Extend voice with corresponding stream
                            for es, ns in best_connections:
                                vm[es].append(prev_contig.streams[ns])

            nix += 1

            # If we have already assigned a voice to all notes in the score,
            # break the loop (or if there are no more neighboring contigs to process)
            if all([note.voice is not None for note in self.notes]) or (nix > len(self) + 1):
                keep_loop = False
