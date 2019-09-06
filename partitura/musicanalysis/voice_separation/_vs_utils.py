#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base classes to represent score elements for voice separation

TODO
----
* Reduce overlap between the main Note and Score classes in partitura
* Finish documentation.
"""
import numpy as np

from collections import defaultdict
from statistics import mode

from ...utils import add_field


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

    def __init__(self, pitch, onset, duration, note_id, velocity=None, voice=None):

        # ID of the note
        self.id = note_id
        self.pitch = pitch
        self.onset = onset
        self.duration = duration
        self.offset = onset + duration

        self.skip_contig = 0

        self.is_grace = self.duration == 0
        self._grace = None
        self._voice = voice
        self.velocity = None

    @property
    def voice(self):
        return self._voice

    @voice.setter
    def voice(self, voice):
        self._voice = voice

        if self._grace is not None:
            self._grace.voice = self._voice

    def is_sounding(self, tp):
        return tp >= self.onset and tp < self.offset

    @property
    def grace(self):
        return self._grace

    @grace.setter
    def grace(self, grace):
        self._grace = grace
        self._grace.voice = self.voice

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

    def __init__(self, notes, rep_pitch='max'):

        if any([n.onset != notes[0].onset for n in notes]):
            raise ValueError('All notes in the chord must have the same onset')
        if any([n.offset != notes[0].offset for n in notes]):
            raise ValueError('All notes in the chord must have the same offset')
        self.notes = notes

        self.onset = self.notes[0].onset
        self.offset = self.notes[0].offset
        self.duration = self.notes[0].duration

        self.velocity = [n.velocity for n in self.notes]

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


class VSScore(VSBaseScore):
    """Class to represent a score for voice separation

    TODO:
    * rename this class or simplify to avoid overlap in naming
      conventions with the main package
    * better handle grace notes
    """

    def __init__(self, score, delete_gracenotes=True):

        # Score
        self.score = score

        # Get the IDs of the notes
        if 'id' not in self.score.dtype.names:
            self.score = add_field(self.score, [('id', int)])
            self.score['id'] = np.arange(len(self.score), dtype=int)

        if delete_gracenotes:
            # TODO: Handle grace notes correctly
            self.score = self.score[score['duration'] != 0]
        else:
            grace_note_idxs = np.where(score['duration'] == 0)[0]

            main_notes_idxs = []
            grace_notes = []
            for g_i in grace_note_idxs:
                grace_note = self.score[g_i]
                same_onset_idxs = np.where(self.score['onset'] == grace_note['onset'])[0]
                same_onset_idxs = same_onset_idxs[same_onset_idxs != g_i]
                candidate_notes = self.score[same_onset_idxs]

                main_notes_idxs.append(same_onset_idxs[
                    np.argmin(abs(candidate_notes['pitch'] - grace_note['pitch']))])

        self.notes = []

        for n in self.score:

            note = VSNote(pitch=n['pitch'],
                          onset=n['onset'],
                          duration=n['duration'],
                          note_id=n['id'],
                          velocity=n['velocity'] if 'velocity' in self.score.dtype.names else None)

            self.notes.append(note)

        # import pdb
        # pdb.set_trace()
        if not delete_gracenotes:
            for g_i, m_i in zip(grace_note_idxs, main_notes_idxs):
                self.notes[g_i].is_grace = True
                self.notes[m_i].grace = self.notes[g_i]

        self.notes = np.array(sort_by_onset(self.notes))

        super(VSScore, self).__init__(self.notes)

        self.contigs = None

    def write_txt(self, outfile, skip_notes_wo_voice=False):

        if skip_notes_wo_voice:
            out_notes = [n for n in self.notes if n.voice is not None]

        else:
            out_notes = self.notes

        out_array = []
        for n in out_notes:
            out_note = (n.pitch, n.onset, n.duration, n.voice if n.voice is not None else -1)
            out_array.append(out_note)

        np.savetxt(outfile, np.array(out_array), delimiter='\t',
                   header='\t'.join(['pitch', 'onset', 'duration', 'voice']),
                   fmt='%.4f')

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

        if len(n_voices) > 5:
            if n_voices[:-3].max() < n_voices[-3:].max():
                self.num_voices = n_voices[:-3].max()
            else:
                self.num_voices = np.max(n_voices)

        else:
            self.num_voices = np.max(n_voices)

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
