#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module cotains tests for exporting MIDI file methods.
"""
import glob
import io
import logging
import random
import time
import warnings
from collections import defaultdict, Counter, OrderedDict
import unittest
from tempfile import TemporaryFile
import mido
import numpy as np
from tempfile import TemporaryDirectory
import os

from partitura import save_score_midi, save_performance_midi, load_performance_midi, load_score
from partitura.utils import partition
import partitura.score as score

from partitura.performance import PerformedPart, Performance
from partitura.io.exportmidi import _HAS_FASTMIDI
from tests import MIDIEXPORT_TESTFILES

LOGGER = logging.getLogger(__name__)

RNG = np.random.RandomState(1984)


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
        t = 0
        for msg in mid.tracks[0]:
            t += msg.time
            if msg.type == "note_on":
                assert t == 0
                break

        mid = export_and_read(part, anacrusis_behavior="pad_bar")
        t = 0
        for msg in mid.tracks[0]:
            t += msg.time
            if msg.type == "note_on":
                assert t == 3, f"Incorrect time of first note on: {t} (should be 3)"
                break

    def test_midi_export_score(self):

        part = score.Part("id")
        # 1 div is 1 quarter
        part.set_quarter_duration(0, 1)
        # 4/4 at t=0
        part.add(score.TimeSignature(4, 4), 0)
        # global tempo
        part.add(score.Tempo(bpm=60), 0)
        # global dynamic (MIDI velocity 60)
        part.add(score.Dynamic(velocity=66.66), 0)

        # ANACRUSIS
        # quarter note from t=0 to t=1
        part.add(score.Note("c", 4), 0, 1)
        # incomplete measure from t=0 to t=1
        part.add(score.Measure(), 0, 1)

        # whole note from t=1 to t=5
        part.add(score.Note("c", 4), 1, 5)
        # add missing measures
        score.add_measures(part)

        scr = score.Score(part)
        mid = export_and_read(scr, anacrusis_behavior="shift")
        t = 0
        for msg in mid.tracks[0]:
            t += msg.time
            if msg.type == "note_on":
                self.assertEqual(t, 0)
                break

        mid = export_and_read(scr, anacrusis_behavior="pad_bar")
        t = 0
        for msg in mid.tracks[0]:
            t += msg.time
            if msg.type == "note_on":
                self.assertEqual(t, 3)
                break

        for msg in mid.tracks[0]:
            if msg.type == "set_tempo":
                self.assertEqual(msg.tempo, 1000000)
            elif msg.type in ("note_on", "note_off"):
                self.assertEqual(msg.velocity, 60)


def n_items_per_part_voice(pg, cls):
    n_items = []
    for part in score.iter_parts(pg):
        n = sum(1 for _ in part.iter_all(cls))
        n_items.extend([n] * len(set(n.voice for n in part.notes_tied)))
    return n_items


def export_and_read_performance(perf_info, **kwargs):

    with TemporaryFile(suffix=".mid") as f:
        save_performance_midi(
            performance_data=perf_info,
            out=f,
            **kwargs,
        )
        f.flush()
        f.seek(0)
        return mido.MidiFile(file=f)


class TestExportPerformanceMIDI(unittest.TestCase):
    def _export_and_read(self, perf_info, **kwargs):
        return export_and_read_performance(perf_info, **kwargs)

    def test_save_single_track(self):

        ppart = generate_random_performance(n_tracks=1)

        note_array = ppart.note_array()

        tracks = note_array["track"]

        self.assertTrue(all(tracks == 0))

        mf_from_ppart = self._export_and_read(ppart)

        self.assertEqual(ppart.num_tracks, len(mf_from_ppart.tracks))

    def test_save_multiple_track(self):
        n_tracks = RNG.randint(2, 10, 10)
        for nt in n_tracks:

            ppart = generate_random_performance(n_notes=10 * nt, n_tracks=nt)

            performance = Performance(performedparts=ppart)
            mf_from_perf = self._export_and_read(performance)
            self.assertEqual(performance.num_tracks, len(mf_from_perf.tracks))


def generate_random_performance(n_notes=100, beat_period=0.5, n_tracks=3):

    note_array = np.empty(
        (n_notes),
        dtype=[
            ("onset_sec", "f4"),
            ("duration_sec", "f4"),
            ("pitch", "i4"),
            ("velocity", "i4"),
            ("track", "i4"),
            ("channel", "i4"),
            ("id", "U256"),
        ],
    )

    note_array["pitch"] = RNG.randint(0, 128, n_notes)

    ioi = RNG.rand(n_notes - 1) * beat_period

    note_array["onset_sec"] = np.r_[0, np.cumsum(ioi)]

    note_array["duration_sec"] = np.clip(
        RNG.rand(n_notes) * 2 * beat_period,
        a_min=0.3,
        a_max=2,
    )

    note_array["velocity"] = RNG.randint(54, 90, n_notes)

    note_array["channel"] *= 0

    note_array["id"] = np.array([f"n{i}" for i in range(n_notes)])

    track_idxs = np.arange(n_notes)
    RNG.shuffle(track_idxs)

    track_length = int(np.floor(n_notes / n_tracks))

    for i in range(n_tracks):
        if i < n_tracks - 1:
            idx = track_idxs[i * track_length : (i + 1) * track_length]
        else:
            idx = track_idxs[i * track_length :]
        note_array["track"][idx] = i

    performed_part = PerformedPart.from_note_array(note_array)
    return performed_part

class TestIncompleteMeasures(unittest.TestCase):
    def test_timesigchange(self):
        # test the behavior with the time_sig_change parameter in midi export
        score_data = load_score(MIDIEXPORT_TESTFILES[0])
        with TemporaryDirectory() as tmpdir:
            temp_midi_path = os.path.join(tmpdir, "temp_midi.mid")
            save_score_midi(score_data, out=temp_midi_path, anacrusis_behavior="time_sig_change", part_voice_assign_mode = 4 )
            mid_pt = mido.MidiFile(temp_midi_path)
        ts_messages = [m for m in list(mid_pt.tracks[0]) if isinstance(m,mido.MetaMessage) and m.type == "time_signature"]
        self.assertTrue(len(ts_messages)==5)
        self.assertTrue(ts_messages[0].numerator == 1)
        self.assertTrue(ts_messages[1].numerator == 4)
        self.assertTrue(ts_messages[2].numerator == 3)
        self.assertTrue(ts_messages[3].numerator == 1)
        self.assertTrue(ts_messages[4].numerator == 4)

    def test_pad_bar(self):
        # test the behavior with the pad_bar parameter in midi export
        score_data = load_score(MIDIEXPORT_TESTFILES[0])
        with TemporaryDirectory() as tmpdir:
            temp_midi_path = os.path.join(tmpdir, "temp_midi.mid")
            save_score_midi(score_data, out=temp_midi_path, anacrusis_behavior="pad_bar", part_voice_assign_mode = 4 )
            mid_pt = mido.MidiFile(temp_midi_path)
        ts_messages = [m for m in list(mid_pt.tracks[0]) if isinstance(m,mido.MetaMessage) and m.type == "time_signature"]
        self.assertTrue(len(ts_messages)==1)
        # check the position of the first note_on message
        all_messages = list(mid_pt.tracks[0])
        cumulative_position = 0
        index = 0
        while not (isinstance(all_messages[index],mido.Message) and all_messages[index].type == "note_on"):
            cumulative_position += all_messages[index].time
            index+=1
        self.assertTrue(cumulative_position==3)


def _dump_midifile_for_comparison(mf):
    """Reduce a mido.MidiFile to (type, ticks_per_beat, per-track lists of
    (absolute tick, message fields)), ignoring end_of_track: minimidi always
    writes its end-of-track marker one tick after the last event, mido at
    the last event, which carries no musical information.
    """
    tracks = []
    for track in mf.tracks:
        t, msgs = 0, []
        for msg in track:
            t += msg.time
            if msg.type == "end_of_track":
                continue
            fields = msg.dict()
            fields.pop("time")
            msgs.append((t, tuple(sorted(fields.items()))))
        tracks.append(msgs)
    return mf.type, mf.ticks_per_beat, tracks


def _make_awkward_performance(seed, with_programs, with_meta):
    """Multi-track/channel performance with controls, signatures, retriggered
    same-pitch notes at a shared tick, and optionally programs / meta_other.
    """
    rng = random.Random(seed)
    parts = []
    for track in range(3):
        notes = []
        for k in range(60):
            on = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) + rng.random() * 3
            if k % 7 == 0:
                notes.append(
                    dict(id=f"a{k}", midi_pitch=60, note_on=1.0, note_off=1.5,
                         track=track, channel=track)
                )
                notes.append(
                    dict(id=f"b{k}", midi_pitch=60, note_on=1.5, note_off=2.0,
                         track=track, channel=track)
                )
            notes.append(
                dict(id=f"n{k}", midi_pitch=rng.randint(30, 100), note_on=on,
                     note_off=on + rng.random(), velocity=rng.randint(1, 127),
                     track=track, channel=rng.randint(0, 3))
            )
        controls = [
            dict(time=rng.random() * 3, time_tick=0, number=64,
                 value=rng.randint(0, 127), track=track, channel=0)
            for _ in range(10)
        ]
        programs = (
            [dict(time=0.0, time_tick=0, program=5 + track, track=track, channel=track)]
            if with_programs
            else []
        )
        time_sigs = (
            [dict(time=0.0, time_tick=0, beats=3, beat_type=4, track=track),
             dict(time=2.0, time_tick=0, beats=6, beat_type=8, track=track)]
            if track == 0
            else []
        )
        key_sigs = (
            [dict(time=0.0, time_tick=0, fifths=-3, mode="minor", track=track),
             dict(time=1.0, time_tick=0, fifths=2, mode="major", track=track)]
            if track == 0
            else []
        )
        meta = (
            [dict(type="track_name", name=f"Track {track}", time=0.0, track=track),
             dict(type="smpte_offset", frame_rate=24, hours=1, minutes=2, seconds=3,
                  frames=4, sub_frames=5, time=0.5, track=track),
             dict(type="marker", text="A", time=1.25, track=track)]
            if with_meta
            else []
        )
        parts.append(
            PerformedPart(notes, controls=controls, programs=programs,
                          time_signatures=time_sigs, key_signatures=key_sigs,
                          meta_other=meta, track=track)
        )
    return Performance(parts)


@unittest.skipUnless(
    _HAS_FASTMIDI,
    "the optional partitura._minimidi extension is not built",
)
class TestSavePerformanceMIDIBackends(unittest.TestCase):
    """backend='minimidi' must write the same events, in the same order, as
    backend='mido'."""

    def _assert_backends_agree(self, perf, **kwargs):
        mf_mido = save_performance_midi(perf, None, backend="mido", **kwargs)
        mf_fast = save_performance_midi(perf, None, backend="minimidi", **kwargs)
        self.assertEqual(
            _dump_midifile_for_comparison(mf_mido),
            _dump_midifile_for_comparison(mf_fast),
        )

    def test_backends_agree_awkward_performances(self):
        for seed in range(3):
            for with_programs in (False, True):
                for with_meta in (False, True):
                    for merge in (False, True):
                        with self.subTest(
                            seed=seed,
                            with_programs=with_programs,
                            with_meta=with_meta,
                            merge_tracks_save=merge,
                        ):
                            perf = _make_awkward_performance(
                                seed, with_programs, with_meta
                            )
                            self._assert_backends_agree(
                                perf, merge_tracks_save=merge
                            )

    def test_backends_agree_custom_ppq_mpq_velocity(self):
        perf = _make_awkward_performance(0, False, False)
        # a note without a velocity must get default_velocity on both
        perf[0].notes[0].pnote_dict.pop("velocity", None)
        self._assert_backends_agree(perf, ppq=96, mpq=400000, default_velocity=90)

    def test_round_trip_all_test_files(self):
        # load every MIDI fixture with each import backend, save with each
        # export backend: the written events must agree.
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        midi_files = sorted(
            set(glob.glob(os.path.join(tests_dir, "**", "*.mid"), recursive=True))
            | set(glob.glob(os.path.join(tests_dir, "**", "*.midi"), recursive=True))
        )
        self.assertGreater(len(midi_files), 0)
        for midi_file in midi_files:
            for load_backend in ("mido", "minimidi"):
                with self.subTest(
                    midi_file=os.path.relpath(midi_file, tests_dir),
                    load_backend=load_backend,
                ):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        perf = load_performance_midi(midi_file, backend=load_backend)
                    self._assert_backends_agree(perf)

    def test_minimidi_output_reloads_to_same_notes(self):
        # non-overlapping notes per (channel, pitch), so reloading re-pairs
        # them unambiguously; times are quantized to ticks on export.
        notes = [
            dict(id=f"n{k}", midi_pitch=30 + k % 60, note_on=0.1 * k,
                 note_off=0.1 * k + 0.05, velocity=40 + k % 80, track=0,
                 channel=k % 3)
            for k in range(200)
        ]
        perf = Performance([PerformedPart(notes)])
        tick = 500000 / 480 / 1e6  # seconds per tick at the default mpq/ppq
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.mid")
            save_performance_midi(perf, path, backend="minimidi")
            reloaded = load_performance_midi(path, backend="mido", quiet=True)

        def key(n):
            return (n["note_on"], n["midi_pitch"], n["channel"])

        got = sorted(reloaded[0].notes, key=key)
        want = sorted(perf[0].notes, key=key)
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertEqual(g["midi_pitch"], w["midi_pitch"])
            self.assertEqual(g["channel"], w["channel"])
            self.assertEqual(g["velocity"], w["velocity"])
            self.assertAlmostEqual(g["note_on"], w["note_on"], delta=tick)
            self.assertAlmostEqual(g["note_off"], w["note_off"], delta=tick)

    def test_out_path_file_like_and_none(self):
        perf = _make_awkward_performance(2, True, True)
        expected = _dump_midifile_for_comparison(
            save_performance_midi(perf, None, backend="mido")
        )
        # out=None keeps returning a mido.MidiFile
        mf = save_performance_midi(perf, None, backend="minimidi")
        self.assertIsInstance(mf, mido.MidiFile)
        self.assertEqual(_dump_midifile_for_comparison(mf), expected)
        # path
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.mid")
            self.assertIsNone(save_performance_midi(perf, path, backend="minimidi"))
            self.assertEqual(
                _dump_midifile_for_comparison(mido.MidiFile(path)), expected
            )
        # file-like
        buf = io.BytesIO()
        save_performance_midi(perf, buf, backend="minimidi")
        buf.seek(0)
        self.assertEqual(
            _dump_midifile_for_comparison(mido.MidiFile(file=buf)), expected
        )

    def test_out_of_range_values_raise(self):
        def perf_with(**note_kwargs):
            note = dict(id="n0", midi_pitch=60, note_on=0.0, note_off=1.0)
            note.update(note_kwargs)
            return PerformedPart([note], validate=False)

        for kwargs in (
            dict(midi_pitch=128),
            dict(midi_pitch=-1),
            dict(velocity=128),
            dict(channel=16),
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    save_performance_midi(
                        perf_with(**kwargs), None, backend="minimidi"
                    )


class TestSavePerformanceMIDIBackendDispatch(unittest.TestCase):
    def setUp(self):
        self.ppart = PerformedPart(
            [dict(id="n0", midi_pitch=60, note_on=0.0, note_off=1.0)]
        )

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            save_performance_midi(self.ppart, None, backend="not-a-backend")

    def test_mido_and_auto_backends_return_midifile(self):
        for backend in ("mido", "auto"):
            mf = save_performance_midi(self.ppart, None, backend=backend)
            self.assertIsInstance(mf, mido.MidiFile)


@unittest.skipUnless(
    _HAS_FASTMIDI,
    "the optional partitura._minimidi extension is not built",
)
class TestSavePerformanceMIDISpeed(unittest.TestCase):
    """The minimidi export backend should be much faster than mido's."""

    # measured ~19-24x on this size; require far less so the test is not
    # flaky on a noisy machine while still catching a regression to
    # "about the same speed".
    MIN_SPEEDUP = 4.0
    N_NOTES = 8000
    N_REPEATS = 3

    def test_minimidi_export_is_faster_than_mido(self):
        rng = random.Random(7)
        notes = []
        t = 0.0
        for k in range(self.N_NOTES):
            dur = rng.choice([0.1, 0.2, 0.4])
            notes.append(
                dict(id=f"n{k}", midi_pitch=rng.randint(30, 100), note_on=t,
                     note_off=t + dur, velocity=70, track=0, channel=0)
            )
            t += rng.choice([0.0, 0.05, 0.1])
        perf = Performance([PerformedPart(notes, validate=False)])

        def best(backend):
            # write to a buffer: with out=None the minimidi path re-parses
            # its bytes with mido to return a MidiFile, which would measure
            # mido's parser rather than the export.
            best_time = float("inf")
            for _ in range(self.N_REPEATS):
                buf = io.BytesIO()
                t0 = time.perf_counter()
                save_performance_midi(perf, buf, backend=backend)
                best_time = min(best_time, time.perf_counter() - t0)
            return best_time

        t_mido, t_fast = best("mido"), best("minimidi")
        self.assertGreater(t_fast, 0.0)
        speedup = t_mido / t_fast
        self.assertGreaterEqual(
            speedup,
            self.MIN_SPEEDUP,
            f"backend='minimidi' should be at least {self.MIN_SPEEDUP}x faster "
            f"than backend='mido' for {self.N_NOTES} notes, but was only "
            f"{speedup:.1f}x ({t_mido:.4f}s vs {t_fast:.4f}s)",
        )
