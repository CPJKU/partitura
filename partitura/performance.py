#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains a lightweight ontology to represent a performance in a
MIDI-like format. A performance is defined at the highest level by a
:class:`~partitura.performance.PerformedPart`. This object contains performed
notes as well as continuous control parameters, such as sustain pedal.

"""

import logging

import numpy as np


LOGGER = logging.getLogger(__name__)

__all__ = ["PerformedPart"]


class PerformedPart(object):
    """Represents a performed part, e.g. all notes and related
    controller/modifiers of one single instrument.

    Performed notes are stored as a list of dictionaries, where each
    dictionary represents a performed note, should have at least the
    keys "note_on", "note_off", the onset and offset times of the note
    in seconds, respectively.

    Continuous controls are also stored as a list of dictionaries,
    where each dictionary represents a control change. Each dictionary
    should have a key "type" (the name of the control, e.g.
    "sustain_pedal", "soft_pedal"), "time" (in seconds), and "value"
    (a number).

    Parameters
    ----------
    notes : list
        A list of dictionaries containing performed note information.
    id : str
        The identifier of the part
    controls : list
        A list of dictionaries containing continuous control information
    part_name : str
        Name for the part
    sustain_pedal_threshold : int
        The threshold above which sustain pedal values are considered
        to be equivalent to on. For values below the threshold the
        sustain pedal is treated as off. Defaults to 64.

    Attributes
    ----------
    notes : list
        A list of dictionaries containing performed note information.
    id : str
        The identifier of the part
    part_name : str
        Name for the part
    controls : list
        A list of dictionaries containing continuous control
        information
    programs : list
        List of dictionaries containing program change information

    """

    def __init__(
        self,
        notes,
        id=None,
        part_name=None,
        controls=None,
        programs=None,
        sustain_pedal_threshold=64,
    ):
        super().__init__()
        self.id = id
        self.part_name = part_name
        self.notes = notes
        self.controls = controls or []
        self.programs = programs or []

        self.sustain_pedal_threshold = sustain_pedal_threshold

    @property
    def sustain_pedal_threshold(self):
        """The threshold value (number) above which sustain pedal values
        are considered to be equivalent to on. For values below the
        threshold the sustain pedal is treated as off. Defaults to 64.

        Based on the control items of type "sustain_pedal", in
        combination with the value of the "sustain_pedal_threshold"
        attribute, the note dictionaries will be extended with a key
        "sound_off". This key represents the time the note will stop
        sounding. When the sustain pedal is off, `sound_off` will
        coincide with `note_off`.  When the sustain pedal is on,
        `sound_off` will equal the earliest time the sustain pedal is
        off after `note_off`. The `sound_off` values of notes will be
        automatically recomputed each time the
        `sustain_pedal_threshold` is set.

        """
        return self._sustain_pedal_threshold

    @sustain_pedal_threshold.setter
    def sustain_pedal_threshold(self, value):
        # """
        # Set the pedal threshold and update the sound_off
        # of the notes
        # """
        self._sustain_pedal_threshold = value
        adjust_offsets_w_sustain(
            self.notes, self.controls, self._sustain_pedal_threshold
        )

    @property
    def note_array(self):
        """Structured array containing performance information.
        The fields are 'id', 'pitch', 'onset_div', 'duration_div',
        'onset_sec', 'duration_sec' and 'velocity'.
        """

        fields = [
            ("onset_sec", "f4"),
            ("duration_sec", "f4"),
            ("pitch", "i4"),
            ("velocity", "i4"),
            ("track", "i4"),
            ("channel", "i4"),
            ("id", "U256"),
        ]
        note_array = []
        for n in self.notes:
            note_on_sec = n["note_on"]
            offset = n.get("sound_off", n["note_off"])
            duration_sec = offset - note_on_sec
            note_array.append(
                (
                    note_on_sec,
                    duration_sec,
                    n["midi_pitch"],
                    n["velocity"],
                    n.get("track", 0),
                    n.get("channel", 1),
                    n["id"],
                )
            )

        return np.array(note_array, dtype=fields)

    @classmethod
    def from_note_array(cls, note_array, id=None, part_name=None):
        """Create an instance of PerformedPart from a note_array.
        Note that this property does not include non-note information (i.e.
        controls such as sustain pedal).
        """
        if "id" not in note_array.dtype.names:
            n_ids = ["n{0}".format(i) for i in range(len(note_array))]
        else:
            n_ids = note_array["id"]

        if "track" not in note_array.dtype.names:
            tracks = np.zeros(len(note_array), dtype=int)
        else:
            tracks = note_array["track"]

        if "channel" not in note_array.dtype.names:
            channels = np.ones(len(note_array), dtype=int)
        else:
            channels = note_array["channel"]

        notes = []
        for nid, note, track, channel in zip(n_ids, note_array, tracks, channels):
            notes.append(
                dict(
                    id=nid,
                    midi_pitch=note["pitch"],
                    note_on=note["onset_sec"],
                    note_off=note["onset_sec"] + note["duration_sec"],
                    sound_off=note["onset_sec"] + note["duration_sec"],
                    track=track,
                    channel=channel,
                    velocity=note["velocity"],
                )
            )

        return cls(id=id, part_name=part_name, notes=notes, controls=None)


def adjust_offsets_w_sustain(notes, controls, threshold=64):
    # get all note offsets
    offs = np.fromiter((n["note_off"] for n in notes), dtype=float)
    first_off = np.min(offs)
    last_off = np.max(offs)

    # Get pedal times
    pedal = np.array(
        [(x["time"], x["value"] > threshold) for x in controls if x["number"] == 64]
    )

    if len(pedal) == 0:
        for note in notes:
            note["sound_off"] = note["note_off"]
        return

    # sort, just in case
    pedal = pedal[np.argsort(pedal[:, 0]), :]

    # reduce the pedal info to just the times where there is a change in pedal state

    pedal = np.vstack(
        (
            (min(pedal[0, 0] - 1, first_off - 1), 0),
            pedal[0, :],
            # if there is an onset before the first pedal info, assume pedal is off
            pedal[np.where(np.diff(pedal[:, 1]) != 0)[0] + 1, :],
            # if there is an offset after the last pedal info, assume pedal is off
            (max(pedal[-1, 0] + 1, last_off + 1), 0),
        )
    )
    last_pedal_change_before_off = np.searchsorted(pedal[:, 0], offs) - 1

    pedal_state_at_off = pedal[last_pedal_change_before_off, 1]
    pedal_down_at_off = pedal_state_at_off == 1
    next_pedal_time = pedal[last_pedal_change_before_off + 1, 0]

    offs[pedal_down_at_off] = next_pedal_time[pedal_down_at_off]

    for offset, note in zip(offs, notes):
        note["sound_off"] = offset
