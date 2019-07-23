#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Voice Separation Algorithm

Elaine Chew and Xiaodan Wu (2006) Separating Voices in Polyphonic Music:
A Contig Mapping Approach.
"""
import numpy as np

from collections import defaultdict
import numpy.ma as ma

try:
    from score_representation import VSScore, VSNote, VoiceManager, Contig, NoteStream
except ModuleNotFoundError:
    from .score_representation import VSScore, VSNote, VoiceManager, Contig, NoteStream


# Maximal cost of a jump (in Chew  and Wu (2006) is 2 ** 31)
MAX_COST = 1000


def pairwise_cost(prev, nxt):
    """
    Compute pairwise cost between two contigs

    Parameters
    ----------
    prev : Contig, VoiceManager or iterable
        VSNotes in the left side of the connection

    nxt : Contig, VoiceManager or iterable
        VSNotes in the right side of the connection

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


class VoSA(VSScore):
    """Voice estimation algorithm
    """

    def __init__(self, score):
        super(VoSA, self).__init__(score)

        self.make_contigs()
        self.estimate_voices()

        # create a NoteStream for each voice
        self.voices = []
        for vn in range(self.num_voices):
            notes_per_voice = [note for note in self.notes if note.voice == vn]
            self.voices.append(NoteStream(notes_per_voice, voice=vn))

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


if __name__ == '__main__':

    import partitura

    fn = './Three-Part_Invention_No_13_(fragment).musicxml'

    xml = partitura.musicxml.xml_to_notearray(fn)

    vsa = VoSA(xml)
    vsa.write_midi('test.mid')
