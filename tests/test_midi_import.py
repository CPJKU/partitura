#!/usr/bin/env python

import logging
from collections import defaultdict, Counter
from operator import itemgetter
import unittest
from tempfile import NamedTemporaryFile
import mido

from partitura import load_midi
from partitura.utils import partition
import partitura.score as score

LOGGER = logging.getLogger(__name__)

def fill_track(track, notes, divs, vel=64):
    onoffs = defaultdict(list)
    for on, off, pitch, ch in notes:
        on = int(divs*on)
        off = int(divs*off)
        onoffs[on].append(('note_on', pitch, ch))
        onoffs[off].append(('note_off', pitch, ch))

    times = sorted(onoffs.keys())
    prev = 0
    for t in times:
        dt = t - prev
        for msg, pitch, ch in onoffs[t]:
            track.append(mido.Message(msg, note=pitch,
                                      velocity=vel, channel=ch,
                                      time=dt))
            dt = 0
        prev = t

def make_assignment_mode_example():
    divs = 10
    mid = mido.MidiFile(ticks_per_beat=divs)
    track_1 = mido.MidiTrack()
    mid.tracks.append(track_1)
    tempo = int(.5*10**6)
    track_1.append(mido.MetaMessage('track_name', name='Track 1'))
    track_1.append(mido.MetaMessage('instrument_name', name='Instrument 1'))
    track_1.append(mido.Message('program_change', channel=0, program=1))
    track_1.append(mido.MetaMessage('set_tempo', tempo=tempo))
    track_1.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))

    track_2 = mido.MidiTrack()
    mid.tracks.append(track_2)
    track_2.append(mido.MetaMessage('track_name', name='Track 2'))
    track_2.append(mido.MetaMessage('instrument_name', name='Instrument 2'))
    track_2.append(mido.Message('program_change', channel=1, program=57))
    track_2.append(mido.MetaMessage('set_tempo', tempo=tempo))
    track_2.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))


    # on off pitch ch, (times in quarter)
    notes_1 = [(0, 1, 76, 0),
               (1, 2, 76, 0),
               (2, 3, 76, 0),
               (0, 4, 69, 2),
               ]
    notes_2 = [(0, 2, 80, 1),
               (2, 4, 69, 1),
               ]
        
    fill_track(track_1, notes_1, divs)
    fill_track(track_2, notes_2, divs)
    return mid

def get_track_voice_numbers(mid):
    tc_counter = Counter()
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'note_on':
                tc_counter.update(((i, msg.channel),))
    return tc_counter
    
class TestMIDIImportModes(unittest.TestCase):

    def setUp(self):
        self.mid = make_assignment_mode_example()
        self.notes_per_tr_ch = get_track_voice_numbers(self.mid)
        self.tmpfile = NamedTemporaryFile(suffix='.mid')
        self.mid.save(self.tmpfile.name)
        
    def test_midi_import_mode_0(self):
        parts = load_midi(self.tmpfile.name, part_voice_assign_mode=0)
        by_track = partition(itemgetter(0), self.notes_per_tr_ch.keys())

        msg = ('Number of parts {} does not equal number of tracks {} while '
               'testing part_voice_assign_mode=0 in load_midi').format(
                   len(parts), len(by_track))
        self.assertEqual(len(parts), len(by_track), msg)

        for part, tr in zip(parts, by_track):

            msg = '{} should be a Part instance but it is not'.format(part)
            self.assertTrue(isinstance(part, score.Part), msg)

            n_track_notes = sum(self.notes_per_tr_ch[tr_ch] for tr_ch in by_track[tr])
            part_notes = part.notes
            n_part_notes = len(part_notes)
            msg = 'Part should have {} notes but it has'.format(n_track_notes, n_part_notes)
            self.assertEqual(n_track_notes, n_part_notes, msg)

            n_ch_notes = [self.notes_per_tr_ch[tr_ch] for tr_ch in by_track[tr]]
            n_voice_notes = [len(vn) for v, vn in
                             partition(lambda x: x, [n.voice for n in part_notes]).items()]
            msg = ('Part voices should have {} respectively, but they have {}'
                   .format(n_ch_notes, n_voice_notes))
            self.assertEqual(n_ch_notes, n_voice_notes, msg)

    def test_midi_import_mode_1(self):
        parts = load_midi(self.tmpfile.name, part_voice_assign_mode=1)
        by_track = partition(itemgetter(0), self.notes_per_tr_ch.keys())
        msg = ('Number of partgroups {} does not equal number of tracks {} while '
               'testing part_voice_assign_mode=0 in load_midi').format(
                   len(parts), len(by_track))
        self.assertEqual(len(parts), len(by_track), msg)

        for part_group, tr in zip(parts, by_track):

            msg = '{} should be a PartGroup instance but it is not'
            self.assertTrue(isinstance(part_group, score.PartGroup), msg)
            n_parts = len(part_group.children)
            n_channels = len(by_track[tr])
            msg = ('PartGroup should have as many parts as there are '
                   'channels in the corresponding track {}, but it has {}'
                   .format(n_channels, n_parts))
            self.assertEqual(n_parts, n_channels, msg)

            for part, tr_ch in zip(part_group.children, by_track[tr]):
                notes_in_track = self.notes_per_tr_ch[tr_ch]
                notes_in_part = len(part.notes)
                msg = 'Part should have {} notes but it has {}'.format(notes_in_track, notes_in_part)
                self.assertEqual(notes_in_part, notes_in_track)
        #     n_track_notes = sum(self.notes_per_tr_ch[tr_ch] for tr_ch in by_track[tr])
        #     part_notes = part.notes
        #     n_part_notes = len(part_notes)
        #     msg = 'Part should have {} notes but it has'.format(n_track_notes, n_part_notes)
        #     self.assertEqual(n_track_notes, n_part_notes, msg)

        #     n_ch_notes = [self.notes_per_tr_ch[tr_ch] for tr_ch in by_track[tr]]
        #     n_voice_notes = [len(vn) for v, vn in
        #                      partition(lambda x: x, [n.voice for n in part_notes]).items()]
        #     msg = ('Part voices should have {} respectively, but they have {}'
        #            .format(n_ch_notes, n_voice_notes))
        #     self.assertEqual(n_ch_notes, n_voice_notes, msg)

    def tearDown(self):
        # remove tmp file
        self.tmpfile = None

