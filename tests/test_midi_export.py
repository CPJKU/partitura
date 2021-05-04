#!/usr/bin/env python

import logging
from collections import defaultdict, Counter, OrderedDict
import unittest
from tempfile import TemporaryFile
import mido

from partitura import save_score_midi
from partitura.utils import partition
import partitura.score as score

LOGGER = logging.getLogger(__name__)


def get_track_voice_numbers(mid):
    counter = Counter()
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == "note_on":
                counter.update(((i, msg.channel),))
    return counter


def get_part_voice_numbers(parts):
    counter = Counter()
    for i, part in enumerate(score.iter_parts(parts)):
        for note in part.notes_tied:
            counter.update(((i, note.voice),))
    return counter


def fill_track(track, notes, divs, vel=64):
    # add on/off events for note specification (`notes`) to `track`
    onoffs = defaultdict(list)
    for on, off, pitch, ch in notes:
        on = int(divs * on)
        off = int(divs * off)
        onoffs[on].append(("note_on", pitch, ch))
        onoffs[off].append(("note_off", pitch, ch))

    times = sorted(onoffs.keys())
    prev = 0
    for t in times:
        dt = t - prev
        for msg, pitch, ch in onoffs[t]:
            track.append(
                mido.Message(msg, note=pitch, velocity=vel, channel=ch, time=dt)
            )
            dt = 0
        prev = t


def make_assignment_mode_example():
    # create a midi file on which to test the assignment modes in load_midi
    part_1 = score.Part("P1")
    part_2 = score.Part("P2")
    part_3 = score.Part("P3")

    part_1.set_quarter_duration(0, 1)
    part_2.set_quarter_duration(0, 2)
    part_3.set_quarter_duration(0, 3)

    part_1.add(score.TimeSignature(4, 4), 0)
    part_1.add(score.Note(step="C", octave=4, voice=1), 0, 1)
    part_1.add(score.Note(step="B", octave=4, voice=2), 0, 2)
    part_1.add(score.Note(step="B", octave=4, voice=2), 2, 4)
    part_1.add(score.Note(step="B", octave=4, voice=2), 5, 6)
    part_1.add(score.Note(step="B", octave=4, voice=3), 7, 10)

    part_2.add(score.TimeSignature(4, 4), 0)
    part_2.add(score.Tempo(80), 0)
    part_2.add(score.Note(step="D", octave=5, voice=1), 0, 1)
    part_2.add(score.Note(step="E", octave=5, voice=2), 1, 2)
    part_2.add(score.Note(step="F", octave=5, voice=2), 2, 3)
    part_2.add(score.Note(step="G", octave=5, voice=2), 3, 4)

    part_3.add(score.TimeSignature(4, 4), 0)
    part_3.add(score.Note(step="G", octave=4, voice=1), 0, 3)

    pg = score.PartGroup(group_name="P1/P2")
    pg.children = [part_1, part_2]
    for p in pg.children:
        p.parent = pg

    return [pg, part_3]


def get_partgroup(part):
    parent = part
    while parent.parent:
        parent = parent.parent
    return parent


def note_voices(part):
    return [note.voice for note in part.notes_tied]


def note_channels(tr):
    return [msg.channel for msg in tr if msg.type == "note_on"]


def n_notes(pg):
    return sum(len(part.notes_tied) for part in score.iter_parts(pg))


def export_and_read(parts, **kwargs):
    with TemporaryFile(suffix=".mid") as f:
        save_score_midi(parts, f, **kwargs)
        f.flush()
        f.seek(0)
        return mido.MidiFile(file=f)


class TestMIDIExportModes(unittest.TestCase):
    def setUp(self):
        self.parts = make_assignment_mode_example()
        # self.targets = get_partgroup_part_voice_numbers(self.parts)
        self.parts_list = list(score.iter_parts(self.parts))

    def _export_and_read(self, mode):
        return export_and_read(self.parts, part_voice_assign_mode=mode)

    def test_midi_export_mode_0(self):
        m = self._export_and_read(mode=0)
        msg = (
            "Number of parts {} does not equal number of tracks {} while "
            "testing part_voice_assign_mode=0 in save_score_midi".format(
                len(self.parts_list), len(m.tracks)
            )
        )
        self.assertEqual(len(self.parts_list), len(m.tracks), msg)

        for part, track in zip(self.parts_list, m.tracks):
            # voices per note in part
            vc = note_voices(part)
            # channels per note in track
            ch = note_channels(track)
            vcp = partition(lambda x: -1 if x is None else x, vc)
            chp = partition(lambda x: x, ch)
            vc_list = [len(vcp[x]) for x in sorted(vcp.keys())]
            ch_list = [len(chp[x]) for x in sorted(chp.keys())]

            msg = (
                "Track channels should have {} notes respectively, "
                "but they have {}".format(vc_list, ch_list)
            )
            self.assertEqual(vc_list, ch_list, msg)

            ts_part = sum(1 for _ in part.iter_all(score.TimeSignature))
            ts_track = sum(1 for e in track if e.type == "time_signature")
            msg = (
                "Track should have {} time signatures respectively, "
                "but has {}".format(ts_part, ts_track)
            )
            self.assertEqual(ts_part, ts_track, msg)

            ks_part = sum(1 for _ in part.iter_all(score.KeySignature))
            ks_track = sum(1 for e in track if e.type == "key_signature")
            msg = (
                "Track should have {} key signatures respectively, "
                "but has {}".format(ks_part, ks_track)
            )
            self.assertEqual(ks_part, ks_track, msg)

    def test_midi_export_mode_1(self):
        m = self._export_and_read(mode=1)
        partgroups = OrderedDict(
            ((get_partgroup(part), True) for part in self.parts_list)
        )
        n_tracks_trg = len(partgroups)
        msg = (
            "Number of parts {} does not equal number of tracks {} while "
            "testing part_voice_assign_mode=1 in save_score_midi".format(
                n_tracks_trg, len(m.tracks)
            )
        )
        self.assertEqual(n_tracks_trg, len(m.tracks), msg)
        for pg, track in zip(partgroups, m.tracks):
            n_notes_in_pg = n_notes(pg)
            n_notes_in_tr = len(note_channels(track))
            msg = "Track should have {} notes, but has {}".format(
                n_notes_in_pg, n_notes_in_tr
            )
            self.assertEqual(n_notes_in_pg, n_notes_in_tr, msg)

    def test_midi_export_mode_2(self):
        m = self._export_and_read(mode=2)
        msg = (
            "Number of tracks {} does not equal 1 while "
            "testing part_voice_assign_mode=2 in save_score_midi".format(len(m.tracks))
        )
        self.assertEqual(1, len(m.tracks), msg)

        n_channels_trg = len(self.parts_list)
        note_ch = note_channels(m.tracks[0])
        by_channel = partition(lambda x: x, note_ch)
        channels = sorted(by_channel.keys())
        n_channels = len(channels)
        msg = (
            "Number of channels {} does not equal {} while "
            "testing part_voice_assign_mode=2 in save_score_midi".format(
                n_channels, n_channels_trg
            )
        )
        self.assertEqual(n_channels_trg, n_channels, msg)
        for part, ch in zip(self.parts_list, channels):
            n_notes_trg = len(part.notes_tied)
            n_notes = len(by_channel[ch])
            msg = (
                "Number of notes in channel {} should be "
                "{} while testing "
                "part_voice_assign_mode=2 in save_score_midi".format(
                    n_notes, n_notes_trg
                )
            )
            self.assertEqual(n_notes_trg, n_notes, msg)

    def test_midi_export_mode_3(self):
        m = self._export_and_read(mode=3)
        msg = (
            "Number of parts {} does not equal number of tracks {} while "
            "testing part_voice_assign_mode=4 in save_score_midi".format(
                len(self.parts_list), len(m.tracks)
            )
        )
        self.assertEqual(len(self.parts_list), len(m.tracks), msg)

        for part, track in zip(self.parts_list, m.tracks):
            # voices per note in part
            n_notes_trg = len(part.notes_tied)
            # channels per note in track
            ch = note_channels(track)
            n_notes = len(ch)
            n_ch = len(set(ch))
            msg = "Track should have 1 channel, " "but has {} channels".format(n_ch)
            self.assertEqual(1, n_ch, msg)

            msg = "Track should have {} notes, " "but has {} notes".format(
                n_notes_trg, n_notes
            )
            self.assertEqual(n_notes_trg, n_notes, msg)

    def test_midi_export_mode_4(self):
        m = self._export_and_read(mode=4)
        msg = (
            "Number of tracks {} does not equal 1 while "
            "testing part_voice_assign_mode=4 in save_score_midi".format(len(m.tracks))
        )
        self.assertEqual(1, len(m.tracks), msg)

        note_ch = note_channels(m.tracks[0])
        n_ch = len(set(note_ch))
        msg = "Track should have 1 channel, " "but has {} channels".format(n_ch)
        self.assertEqual(1, n_ch, msg)
        n_notes_trg = sum(len(part.notes_tied) for part in score.iter_parts(self.parts))
        n_notes = len(note_ch)
        msg = "Track should have {} notes, " "but has {} notes".format(
            n_notes_trg, n_notes
        )

    def test_midi_export_mode_5(self):
        m = self._export_and_read(mode=5)
        notes_per_tr_ch = get_track_voice_numbers(m)
        notes_per_prt_vc = get_part_voice_numbers(self.parts)
        msg = (
            "Number of tracks {} should equal {} while "
            "testing part_voice_assign_mode=5 in save_score_midi".format(
                len(notes_per_tr_ch), len(notes_per_prt_vc)
            )
        )
        self.assertEqual(len(notes_per_tr_ch), len(notes_per_prt_vc), msg)
        for n_tr_ch, n_prt_vc in zip(
            notes_per_tr_ch.values(), notes_per_prt_vc.values()
        ):
            msg = (
                "Number of notes in track {} should "
                "equal {} in while testing "
                "part_voice_assign_mode=2 in save_score_midi".format(n_tr_ch, n_prt_vc)
            )
            self.assertEqual(n_tr_ch, n_prt_vc, msg)

        ts_part = n_items_per_part_voice(self.parts, score.TimeSignature)
        ts_track = [
            sum(1 for e in track if e.type == "time_signature") for track in m.tracks
        ]
        msg = (
            "Number of time signatures per track should be {} respectively, "
            "but is {}".format(ts_part, ts_track)
        )
        self.assertEqual(ts_part, ts_track, msg)

        ks_part = n_items_per_part_voice(self.parts, score.KeySignature)
        ks_track = [
            sum(1 for e in track if e.type == "key_signature") for track in m.tracks
        ]
        msg = (
            "Number of key signatures per track should be {} respectively, "
            "but is {}".format(ks_part, ks_track)
        )
        self.assertEqual(ks_part, ks_track, msg)

    def test_midi_export_anacrusis(self):
        part = score.Part("id")
        # 1 div is 1 quarter
        part.set_quarter_duration(0, 1)
        # 4/4 at t=0
        part.add(score.TimeSignature(4, 4), 0)

        # ANACRUSIS
        # quarter note from t=0 to t=1
        part.add(score.Note("c", 4), 0, 1)
        # incomplete measure from t=0 to t=1
        part.add(score.Measure(), 0, 1)

        # whole note from t=1 to t=5
        part.add(score.Note("c", 4), 1, 5)
        # add missing measures
        score.add_measures(part)

        # print(part.pretty())

        mid = export_and_read(part, anacrusis_behavior="shift")
        for msg in mid.tracks[0]:
            if msg.type == "note_on":
                assert msg.time == 0
                break

        mid = export_and_read(part, anacrusis_behavior="pad_bar")
        for msg in mid.tracks[0]:
            if msg.type == "note_on":
                assert (
                    msg.time == 3
                ), f"Incorrect time of first note on: {msg.time} (should be 3)"
                break


def n_items_per_part_voice(pg, cls):
    n_items = []
    for part in score.iter_parts(pg):
        n = sum(1 for _ in part.iter_all(cls))
        n_items.extend([n] * len(set(n.voice for n in part.notes_tied)))
    return n_items
