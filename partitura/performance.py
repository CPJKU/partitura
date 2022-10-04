#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains a lightweight ontology to represent a performance in a
MIDI-like format. A performance is defined at the highest level by a
:class:`~partitura.performance.PerformedPart`. This object contains performed
notes as well as continuous control parameters, such as sustain pedal.

"""


from typing import Union, List, Optional, Iterator, Iterable as Itertype
import numpy as np
from partitura.utils import note_array_from_part_list

__all__ = [
    "PerformedPart",
    "Performance",
]


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
        notes: List[dict],
        id: str = None,
        part_name: str = None,
        controls: List[dict] = None,
        programs: List[dict] = None,
        sustain_pedal_threshold: int = 64,
    ) -> None:
        super().__init__()
        self.id = id
        self.part_name = part_name
        self.notes = notes
        self.controls = controls or []
        self.programs = programs or []

        self.sustain_pedal_threshold = sustain_pedal_threshold

    @property
    def sustain_pedal_threshold(self) -> int:
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
    def sustain_pedal_threshold(self, value: int) -> None:
        # """
        # Set the pedal threshold and update the sound_off
        # of the notes
        # """
        self._sustain_pedal_threshold = value
        adjust_offsets_w_sustain(
            self.notes, self.controls, self._sustain_pedal_threshold
        )

    @property
    def num_tracks(self) -> int:
        """Number of tracks"""
        return len(
            set(
                [n.get("track", -1) for n in self.notes]
                + [c.get("track", -1) for c in self.controls]
                + [p.get("track", -1) for p in self.programs]
            )
        )

    def note_array(self, *args, **kwargs) -> np.ndarray:
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
    def from_note_array(
        cls,
        note_array: np.ndarray,
        id: str = None,
        part_name: str = None,
    ):
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


def adjust_offsets_w_sustain(
    notes: List[dict],
    controls: List[dict],
    threshold=64,
) -> None:
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


class Performance(object):
    """Main object for representing a performance.

    The `Performance` object is basically an iterable that provides access to all
    `PerformedPart` objects in a musical score.

    Parameters
    ----------
    id : str
        The identifier of the performance.
    performer: str, optional.
        The person or machine performing.
    title: str, optional
        Title of the score.
    subtitle: str, optional
        Subtitle of the score.
    composer: str, optional
        Composer of the score.
    lyricist: str, optional
        Lyricist of the score.
    copyright: str, optional.
        Copyright notice of the score.

    Attributes
    ----------
    id : str
        See parameters.
    performer: str
        See parameters.
    title: str
        See parameters.
    subtitle: str
        See parameters.
    composer: str
        See parameters.
    lyricist: str
        See parameters.
    copyright: str.
        See parameters.
    """

    id: Optional[str]
    title: Optional[str]
    subtitle: Optional[str]
    lyricist: Optional[str]
    copyright: Optional[str]
    performedparts: List[PerformedPart]

    def __init__(
        self,
        performedparts: Union[PerformedPart, Itertype[PerformedPart]],
        id: str = None,
        performer: Optional[str] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        composer: Optional[str] = None,
        lyricist: Optional[str] = None,
        copyright: Optional[str] = None,
        ensure_unique_tracks: bool = True,
    ) -> None:
        self.id = id

        if isinstance(performedparts, PerformedPart):
            self.performedparts = [performedparts]
        elif isinstance(performedparts, Itertype):

            if not all([isinstance(pp, PerformedPart) for pp in performedparts]):
                raise ValueError(
                    "`performedparts` should be a list of  `PerformedPart` objects!"
                )
            self.performedparts = list(performedparts)
        else:
            raise ValueError(
                "`performedparts` should be a `PerformedPart` or a list of "
                f"`PerformedPart` objects but is {type(performedparts)}."
            )

        # Metadata
        self.performer = performer
        self.title = title
        self.subtitle = subtitle
        self.composer = composer
        self.lyricist = lyricist
        self.copyright = copyright

        if ensure_unique_tracks:
            self.sanitize_track_numbers()

    @property
    def num_tracks(self) -> int:
        """
        Number of tracks in the performance
        """
        n_tracks = len(
            set(
                [(i, n.get("track", -1)) for i, pp in enumerate(self) for n in pp.notes]
                + [
                    (i, c.get("track", -1))
                    for i, pp in enumerate(self)
                    for c in pp.controls
                ]
                + [
                    (i, p.get("track", -1))
                    for i, pp in enumerate(self)
                    for p in pp.programs
                ]
            )
        )

        return n_tracks

    def sanitize_track_numbers(self) -> None:
        """
        Ensure that the track number info in each `PerformedPart` in
        self.performedparts is unique (i.e., that a track number does not appear
        in multiple `PerformedPart` instances)
        """
        unique_track_ids = list(
            set(
                [(i, n.get("track", -1)) for i, pp in enumerate(self) for n in pp.notes]
                + [
                    (i, c.get("track", -1))
                    for i, pp in enumerate(self)
                    for c in pp.controls
                ]
                + [
                    (i, p.get("track", -1))
                    for i, pp in enumerate(self)
                    for p in pp.programs
                ]
            )
        )

        track_map = dict([(tid, ti) for ti, tid in enumerate(unique_track_ids)])

        for i, ppart in enumerate(self):

            for note in ppart.notes:

                note["track"] = track_map[(i, note.get("track", -1))]

            for control in ppart.controls:
                control["track"] = track_map[(i, control.get("track", -1))]

            for program in ppart.programs:
                program["track"] = track_map[(i, program.get("track", -1))]

    def __getitem__(self, index: int) -> PerformedPart:
        """Get `Part in the score by index"""
        return self.performedparts[index]

    def __setitem__(self, index: int, pp: PerformedPart) -> None:
        """Set `Part` in the score by index"""
        # TODO: How to update the score structure as well?
        self.performedparts[index] = pp

    def __iter__(self) -> Iterator[PerformedPart]:
        self.iter_idx = 0
        return self

    def __next__(self) -> PerformedPart:
        if self.iter_idx == len(self.performedparts):
            raise StopIteration
        res = self[self.iter_idx]
        self.iter_idx += 1
        return res

    def __len__(self) -> int:
        """
        The lenght of the score is the number of part objects in `self.parts`
        """
        return len(self.performedparts)

    def note_array(self, *args, **kwargs) -> np.ndarray:
        """
        Get a note array that concatenates the note arrays of all Part/PartGroup
        objects in the score.
        """
        return note_array_from_part_list(self.performedparts, *args, **kwargs)


# Alias for typing performance-like objects
PerformanceLike = Union[List[PerformedPart], PerformedPart, Performance]
