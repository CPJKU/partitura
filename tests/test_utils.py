import unittest
import partitura
import numpy as np

from partitura.utils import music
from tests import MATCH_IMPORT_EXPORT_TESTFILES, VOSA_TESTFILES, MOZART_VARIATION_FILES


RNG = np.random.RandomState(1984)


class TestGetMatchedNotes(unittest.TestCase):
    def test_get_matched_notes(self):
        for fn in MATCH_IMPORT_EXPORT_TESTFILES:
            perf, alignment, scr = partitura.load_match(
                filename=fn,
                create_score=True,
            )
            perf_note_array = perf.note_array()
            scr_note_array = scr.note_array()
            matched_idxs = music.get_matched_notes(
                spart_note_array=scr_note_array,
                ppart_note_array=perf_note_array,
                alignment=alignment,
            )
            scr_pitch = scr_note_array["pitch"][matched_idxs[:, 0]]
            perf_pitch = perf_note_array["pitch"][matched_idxs[:, 1]]

            self.assertTrue(np.all(scr_pitch == perf_pitch))


class TestGetTimeMapsFromAlignment(unittest.TestCase):
    def test_get_time_maps_from_alignment(self):
        for fn in VOSA_TESTFILES:
            scr = partitura.load_musicxml(fn)
            note_ids = scr.note_array()["id"]
            beats_per_minute = 60 / RNG.uniform(0.3, 3, size=2)

            for bpm in beats_per_minute:
                ppart = music.performance_from_part(part=scr[0], bpm=bpm)
                alignment = [
                    dict(label="match", score_id=sid, performance_id=sid)
                    for sid in note_ids
                ]

                (
                    ptime_to_stime_map,
                    stime_to_ptime_map,
                ) = music.get_time_maps_from_alignment(
                    spart_or_note_array=scr[0],
                    ppart_or_note_array=ppart,
                    alignment=alignment,
                    remove_ornaments=True,
                )

                score_onsets = np.arange(4, 0.5)
                performed_onsets = 60 / bpm * score_onsets

                self.assertTrue(
                    np.all(score_onsets == ptime_to_stime_map(performed_onsets))
                )
                self.assertTrue(
                    np.all(performed_onsets == stime_to_ptime_map(score_onsets))
                )


class TestPerformanceFromPart(unittest.TestCase):
    def test_performance_from_part(self):
        for fn in VOSA_TESTFILES:
            scr = partitura.load_musicxml(fn)
            beats_per_minute = 60 / RNG.uniform(0.3, 3, size=2)
            midi_velocity = RNG.randint(30, 127, size=2)
            for bpm, vel in zip(beats_per_minute, midi_velocity):
                ppart = music.performance_from_part(part=scr[0], bpm=bpm, velocity=vel)

                # assert that both objects have the same number of notes
                self.assertEqual(len(scr[0].notes_tied), len(ppart.notes))

                snote_array = scr[0].note_array()
                pnote_array = ppart.note_array()

                # check MIDI velocities
                self.assertTrue(np.all(pnote_array["velocity"] == vel))

                alignment = [
                    dict(label="match", score_id=sid, performance_id=sid)
                    for sid in snote_array["id"]
                ]

                matched_idxs = music.get_matched_notes(
                    spart_note_array=snote_array,
                    ppart_note_array=pnote_array,
                    alignment=alignment,
                )

                # check pitch
                self.assertTrue(
                    np.all(
                        pnote_array["pitch"][matched_idxs[:, 1]]
                        == snote_array["pitch"][matched_idxs[:, 0]]
                    )
                )
                # check durations
                self.assertTrue(
                    np.allclose(
                        pnote_array["duration_sec"],
                        snote_array["duration_beat"] * (60 / bpm),
                    )
                )

                pnote_array = pnote_array[matched_idxs[:, 1]]
                snote_array = snote_array[matched_idxs[:, 0]]

                unique_onsets = np.unique(snote_array["onset_beat"])
                unique_onset_idxs = [
                    np.where(snote_array["onset_beat"] == uo)[0] for uo in unique_onsets
                ]

                # check performed onsets
                perf_onsets = np.array(
                    [
                        np.mean(pnote_array["onset_sec"][uix])
                        for uix in unique_onset_idxs
                    ]
                )

                beat_period = np.diff(perf_onsets) / np.diff(unique_onsets)

                # check that that the performance corresponds to the expected tempo
                self.assertTrue(np.allclose(60 / beat_period, bpm))

    def get_tempo_curve(self, score_onsets, performance_onsets):
        """
        Get tempo curve
        """
        unique_sonsets = np.unique(score_onsets)
        # Ensure that everything is sorted (I'm just paranoid ;)
        unique_sonsets.sort()
        unique_ponsets = np.unique(performance_onsets)
        # Ensure that everything is sorted
        unique_ponsets.sort()

        bp = np.diff(unique_ponsets) / np.diff(unique_sonsets)

        # Beats per minute for each of the unique onsets
        # the last bpm is just assuming that the tempo remains
        # constant after the last onset.
        bpm = np.r_[60 / bp, 60 / bp[-1]]

        return bpm

    def test_performance_notearray_from_score_notearray_bpm(self):
        """
        Test possibilities for bpm argument in
        utils.music.performance_notearray_from_score_notearray
        """
        score = partitura.load_score(MOZART_VARIATION_FILES["musicxml"])

        score_note_array = score.note_array()

        unique_onsets = np.unique(score_note_array["onset_beat"])
        unique_onsets.sort()
        # Test constant tempo
        bpm = 30
        velocity = 65
        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            bpm=bpm,
            velocity=velocity,
        )

        self.assertTrue(
            np.allclose(
                self.get_tempo_curve(
                    score_note_array["onset_beat"],
                    perf_note_array["onset_sec"],
                ),
                bpm,
            )
        )

        # Test callable tempo
        def bpm_fun(onset):
            """
            Test function the first half of the piece will be played
            twice as fast
            """
            if isinstance(onset, (int, float)):
                onset = np.array([onset])

            bpm = np.zeros(len(onset), dtype=float)

            midpoint = (unique_onsets.max() - unique_onsets.min()) / 2
            bpm[np.where(onset <= midpoint)[0]] = 120
            bpm[np.where(onset > midpoint)[0]] = 60

            return bpm

        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            bpm=bpm_fun,
            velocity=velocity,
        )

        bpm = self.get_tempo_curve(
            score_note_array["onset_beat"],
            perf_note_array["onset_sec"],
        )

        midpoint = (unique_onsets.max() - unique_onsets.min()) / 2

        self.assertTrue(
            np.allclose(
                bpm[np.where(unique_onsets <= midpoint)[0]],
                120,
            )
        )

        self.assertTrue(np.allclose(bpm[np.where(unique_onsets > midpoint)[0]], 60))

        # Test tempo as an array
        bpm_expected = 40 * RNG.rand(len(unique_onsets)) + 30

        # Test using 1d array
        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            bpm=np.column_stack((unique_onsets, bpm_expected)),
            velocity=velocity,
        )

        bpm_predicted = self.get_tempo_curve(
            score_note_array["onset_beat"],
            perf_note_array["onset_sec"],
        )

        # do not consider the last element, since get_tempo_curve only computes
        # the tempo up to the last onset (otherwise offsets need to be considered)
        self.assertTrue(np.allclose(bpm_expected[:-1], bpm_predicted[:-1], atol=1e-3))

        try:
            # This should trigger an error because bpm_expected is a 1D array
            perf_note_array = music.performance_notearray_from_score_notearray(
                snote_array=score_note_array,
                bpm=bpm_expected,
                velocity=velocity,
            )
            self.assertTrue(False)

        except ValueError:
            # We are expecting the previous code to trigger an error
            self.assertTrue(True)

    def get_velocity_curves(self, velocity, score_onsets):
        """
        Get velocity curve by aggregating MIDI velocity values for
        each onset
        """
        unique_onsets = np.unique(score_onsets)
        # Ensure that everything is sorted (I'm just paranoid ;)
        unique_onsets.sort()

        unique_onset_idxs = [np.where(score_onsets == uo)[0] for uo in unique_onsets]
        velocity_curve = np.array([velocity[ui].mean() for ui in unique_onset_idxs])

        return velocity_curve

    def test_performance_notearray_from_score_notearray_velocity(self):
        """
        Test velocity arguments in
        utils.music.performance_notearray_from_score_notearray
        """
        score = partitura.load_score(MOZART_VARIATION_FILES["musicxml"])

        score_note_array = score.note_array()

        unique_onsets = np.unique(score_note_array["onset_beat"])
        unique_onsets.sort()

        # Test constant velocity
        bpm = 120
        velocity = 65
        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            bpm=bpm,
            velocity=velocity,
        )

        self.assertTrue(all(perf_note_array["velocity"] == velocity))

        # Test callable velocity
        def vel_fun(onset):
            """
            Test function the first half of the piece will be played
            twice as loud
            """
            if isinstance(onset, (int, float)):
                onset = np.array([onset])

            vel = np.zeros(len(onset), dtype=float)

            midpoint = (unique_onsets.max() - unique_onsets.min()) / 2
            vel[np.where(onset <= midpoint)[0]] = 120
            vel[np.where(onset > midpoint)[0]] = 60

            return vel

        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            bpm=bpm,
            velocity=vel_fun,
        )

        vel = self.get_velocity_curves(
            perf_note_array["velocity"], score_note_array["onset_beat"]
        )

        midpoint = (unique_onsets.max() - unique_onsets.min()) / 2

        self.assertTrue(
            np.allclose(
                vel[np.where(unique_onsets <= midpoint)[0]],
                120,
            )
        )

        self.assertTrue(np.allclose(vel[np.where(unique_onsets > midpoint)[0]], 60))

        # Test tempo as an array
        vel_expected = np.round(40 * RNG.rand(len(unique_onsets)) + 30)

        # Test using 1d array
        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            velocity=np.column_stack((unique_onsets, vel_expected)),
            bpm=bpm,
        )

        vel_predicted = self.get_velocity_curves(
            perf_note_array["velocity"],
            score_note_array["onset_beat"],
        )

        self.assertTrue(np.allclose(vel_expected, vel_predicted, atol=1e-3))

        try:
            # This should trigger an error because vel_expected is a 1D array
            perf_note_array = music.performance_notearray_from_score_notearray(
                snote_array=score_note_array,
                bpm=bpm,
                velocity=vel_expected,
            )
            self.assertTrue(False)

        except ValueError:
            # We are expecting the previous code to trigger an error
            self.assertTrue(True)
