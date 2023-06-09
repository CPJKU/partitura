#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing matchfiles
"""
import os
from typing import Union, Tuple, Optional, Callable, List
import warnings

import numpy as np

from partitura import score
from partitura.score import Part, Score
from partitura.performance import PerformedPart, Performance
from partitura.musicanalysis import estimate_voices, estimate_key

from partitura.io.matchlines_v0 import (
    FROM_MATCHLINE_METHODS as FROM_MATCHLINE_METHODSV0,
    parse_matchline as parse_matchlinev0,
    MatchInfo as MatchInfoV0,
    MatchMeta as MatchMetaV0,
    MatchSnote as MatchSnoteV0,
    MatchNote as MatchNoteV0,
    MatchSnoteNote as MatchSnoteNoteV0,
    MatchSnoteDeletion as MatchSnoteDeletionV0,
    MatchSnoteTrailingScore as MatchSnoteTrailingScoreV0,
    MatchInsertionNote as MatchInsertionNoteV0,
    MatchHammerBounceNote as MatchHammerBounceNoteV0,
    MatchTrailingPlayedNote as MatchTrailingPlayedNoteV0,
    MatchSustainPedal as MatchSustainPedalV0,
    MatchSoftPedal as MatchSoftPedalV0,
    MatchTrillNote as MatchTrillNoteV0,
)

from partitura.io.matchlines_v1 import (
    FROM_MATCHLINE_METHODS as FROM_MATCHLINE_METHODSV1,
    MatchInfo as MatchInfoV1,
    MatchScoreProp as MatchScorePropV1,
    MatchSection as MatchSectionV1,
    MatchStime as MatchStimeV1,
    MatchPtime as MatchPtimeV1,
    MatchStimePtime as MatchStimePtimeV1,
    MatchSnote as MatchSnoteV1,
    MatchNote as MatchNoteV1,
    MatchSnoteNote as MatchSnoteNoteV1,
    MatchSnoteDeletion as MatchSnoteDeletionV1,
    MatchInsertionNote as MatchInsertionNoteV1,
    MatchSustainPedal as MatchSustainPedalV1,
    MatchSoftPedal as MatchSoftPedalV1,
    MatchOrnamentNote as MatchOrnamentNoteV1,
)

from partitura.io.matchfile_base import (
    MatchError,
    MatchFile,
    MatchLine,
    BaseSnoteLine,
    BaseSnoteNoteLine,
    BaseStimePtimeLine,
    BaseDeletionLine,
    BaseInsertionLine,
    BaseOrnamentLine,
    BaseSustainPedalLine,
    BaseSoftPedalLine,
)

from partitura.io.matchfile_utils import (
    Version,
    number_pattern,
    MatchTimeSignature,
    MatchKeySignature,
    format_pnote_id,
)

from partitura.utils.music import (
    midi_ticks_to_seconds,
    pitch_spelling_to_midi_pitch,
    ensure_pitch_spelling_format,
    key_name_to_fifths_mode,
    estimate_clef_properties,
    note_array_from_note_list,
)


from partitura.utils.misc import (
    deprecated_alias,
    deprecated_parameter,
    PathLike,
    get_document_name,
)

from partitura.utils.generic import interp1d, partition, iter_current_next

__all__ = ["load_match"]


def get_version(line: str) -> Version:
    """
    Get version from the first line. Since the
    first version of the format did not include this line,
    we assume that the version is 0.1.0 if no version is
    found.

    Parameters
    ----------
    line: str
        The first line of the match file.

    Returns
    -------
    version : Version
        The version of the match file
    """
    version = Version(0, 1, 0)

    for parser in (MatchInfoV1, MatchInfoV0):
        try:
            ml = parser.from_matchline(line)
            if isinstance(getattr(ml, "Value", None), Version):
                version = ml.Value
                return version

        except MatchError:

            pass

    return version


def parse_matchline(
    line: str,
    from_matchline_methods: List[Callable[[str], MatchLine]],
    version: Version,
    debug: bool = False,
) -> Optional[MatchLine]:
    """
    Return objects representing the line as one of:

    * hammer_bounce-PlayedNote.
    * info(Attribute, Value).
    * insertion-PlayedNote.
    * ornament(Anchor)-PlayedNote.
    * ScoreNote-deletion.
    * ScoreNote-PlayedNote.
    * ScoreNote-trailing_score_note.
    * trailing_played_note-PlayedNote.
    * trill(Anchor)-PlayedNote.
    * meta(Attribute, Value, Bar, Beat).
    * sustain(Time, Value)
    * soft(Time, Value)

    or None if none can be matched

    Parameters
    ----------
    line : str
        Line of the match file
    from_matchline_methods : List[Callable[[str], MatchLine]]

    Returns
    -------
    matchline : subclass of `MatchLine`
       Object representing the line.
    """

    matchline = None
    for from_matchline in from_matchline_methods:
        try:
            matchline = from_matchline(line, version=version)
            break
        except Exception as e:
            if not isinstance(e, MatchError):
                print(line, e, version)  # pragma: no cover
            continue

    return matchline


@deprecated_alias(fn="filename", create_part="create_score")
def load_matchfile(
    filename: PathLike,
) -> MatchFile:
    """
    Load a Matchfile as a `MatchFile` instance
    """

    if not os.path.exists(filename):
        raise ValueError("Filename does not exist")  # pragma: no cover

    with open(filename) as f:
        raw_lines = f.read().splitlines()

    version = get_version(raw_lines[0])

    from_matchline_methods = FROM_MATCHLINE_METHODSV1
    if version < Version(1, 0, 0):
        from_matchline_methods = FROM_MATCHLINE_METHODSV0

    parsed_lines = [
        parse_matchline(line, from_matchline_methods, version) for line in raw_lines
    ]

    parsed_lines = [pl for pl in parsed_lines if pl is not None]

    mf = MatchFile(lines=parsed_lines)

    return mf


@deprecated_alias(fn="filename", create_part="create_score")
def load_match(
    filename: PathLike,
    create_score: bool = False,
    pedal_threshold: int = 64,
    first_note_at_zero: bool = False,
    offset_duration_whole: bool = True,
) -> Tuple[Union[Performance, list, Score]]:
    """
    Load a matchfile.

    Parameters
    ----------
    filename : str
        The matchfile
    create_score : bool, optional
        When True create a Part object from the snote information in
        the match file. Defaults to False.
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.

    Returns
    -------
    performance : :class:partitura.performance.Performance
    alignment : list
        The score--performance alignment, a list of dictionaries
    scr : :class:partitura.score.Score
        The score. This item is only returned when `create_score` = True.
    """
    # Parse Matchfile
    mf = load_matchfile(filename)

    # Generate PerformedPart
    ppart = performed_part_from_match(mf, pedal_threshold, first_note_at_zero)

    performance = Performance(
        id=get_document_name(filename),
        performedparts=ppart,
    )
    # Generate Part
    if create_score:
        spart = part_from_matchfile(
            mf,
            match_offset_duration_in_whole=offset_duration_whole,
        )

        scr = score.Score(id=get_document_name(filename), partlist=[spart])
    # Alignment
    alignment = alignment_from_matchfile(mf)

    if create_score:
        return performance, alignment, scr
    else:
        return performance, alignment


def note_alignment_from_matchfile(mf: MatchFile) -> List[dict]:
    """
    Get a note-level alignment from a MatchFile instance

    Parameters
    ----------
    mf : MatchFile
        A score-to-performance alignment

    Returns
    -------
    results : List[dict]
        An alignmnet as a list of dictionaries for each note.
    """
    result = []

    for line in mf.lines:
        if isinstance(line, BaseSnoteNoteLine):
            result.append(
                dict(
                    label="match",
                    score_id=str(line.snote.Anchor),
                    performance_id=format_pnote_id(line.note.Id),
                )
            )

        elif isinstance(
            line,
            BaseDeletionLine,
        ):
            if "leftOutTied" in line.snote.ScoreAttributesList:
                continue
            else:
                result.append(dict(label="deletion", score_id=str(line.snote.Anchor)))
        elif isinstance(
            line,
            BaseInsertionLine,
        ):
            result.append(
                dict(label="insertion", performance_id=format_pnote_id(line.note.Id))
            )
        elif isinstance(line, BaseOrnamentLine):
            if isinstance(line, MatchTrillNoteV0):
                ornament_type = "trill"
            elif isinstance(line, MatchOrnamentNoteV1):
                ornament_type = line.OrnamentType
            else:
                ornament_type = "generic_ornament"
            result.append(
                dict(
                    label="ornament",
                    score_id=str(line.Anchor),
                    performance_id=format_pnote_id(line.note.Id),
                    type=ornament_type,
                )
            )

    return result


# alias
alignment_from_matchfile = note_alignment_from_matchfile


def performed_part_from_match(
    mf: MatchFile,
    pedal_threshold: int = 64,
    first_note_at_zero: bool = False,
) -> PerformedPart:
    """
    Make PerformedPart from performance info in a MatchFile

    Parameters
    ----------
    mf : MatchFile
        A MatchFile instance
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.

    Returns
    -------
    ppart : PerformedPart
        A performed part

    """
    # Get midi time units
    mpq = mf.info("midiClockRate")  # 500000 -> microseconds per quarter
    ppq = mf.info("midiClockUnits")  # 500 -> parts per quarter

    # PerformedNote instances for all MatchNotes
    notes = []

    first_note = next(mf.iter_notes(), None)
    if first_note and first_note_at_zero:
        offset = midi_ticks_to_seconds(first_note.Onset, mpq=mpq, ppq=ppq)
        offset_tick = first_note.Onset
    else:
        offset = 0
        offset_tick = 0

    notes = [
        dict(
            id=format_pnote_id(note.Id),
            midi_pitch=note.MidiPitch,
            note_on=midi_ticks_to_seconds(note.Onset, mpq, ppq) - offset,
            note_off=midi_ticks_to_seconds(note.Offset, mpq, ppq) - offset,
            note_on_tick=note.Onset - offset_tick,
            note_off_tick=note.Offset - offset_tick,
            sound_off=midi_ticks_to_seconds(note.Offset, mpq, ppq) - offset,
            velocity=note.Velocity,
            track=getattr(note, "Track", 0),
            channel=getattr(note, "Channel", 1),
        )
        for note in mf.notes
    ]

    # SustainPedal instances for sustain pedal lines
    sustain_pedal = [
        dict(
            number=64,
            time=midi_ticks_to_seconds(ped.Time, mpq, ppq),
            value=ped.Value,
        )
        for ped in mf.sustain_pedal
    ]

    # SoftPedal instances for soft pedal lines
    soft_pedal = [
        dict(
            number=67,
            time=midi_ticks_to_seconds(ped.Time, mpq, ppq),
            value=ped.Value,
        )
        for ped in mf.soft_pedal
    ]

    # Make performed part
    ppart = PerformedPart(
        id="P1",
        part_name=mf.info("piece"),
        notes=notes,
        controls=sustain_pedal + soft_pedal,
        sustain_pedal_threshold=pedal_threshold,
    )
    return ppart


def sort_snotes(snotes: List[BaseSnoteLine]) -> List[BaseSnoteLine]:
    """
    Sort s(core)notes.

    Parameters
    ----------
    snotes : list
        The score notes

    Returns
    -------
    snotes_sorted : list
        The sorted score notes
    """
    sidx = np.lexsort(
        list(zip(*[(float(n.Offset), float(n.Beat), float(n.Measure)) for n in snotes]))
    )
    return [snotes[i] for i in sidx if snotes[i].NoteName.lower() != "r"]


def part_from_matchfile(
    mf: MatchFile,
    match_offset_duration_in_whole: bool = True,
) -> Part:
    """
    Create a score part from a matchfile.

    Parameters
    ----------
    mf : MatchFile
        An instance of `MatchFile`

    match_offset_duration_in_whole: Boolean
        A flag for the type of offset and duration given in the matchfile.
        When true, the function expects the values to be given in whole
        notes (e.g. 1/4 for a quarter note) independet of time signature.


    Returns
    -------
    part : partitura.score.Part
        An instance of `Part` containing score information.

    """
    part = score.Part("P1", mf.info("piece"))
    snotes = sort_snotes(mf.snotes)

    ts = mf.time_signatures
    min_time = snotes[0].OnsetInBeats  # sorted by OnsetInBeats
    max_time = max(n.OffsetInBeats for n in snotes)
    _, beats_map, _, beat_type_map, min_time_q, max_time_q = make_timesig_maps(
        ts, max_time
    )

    # compute necessary divs based on the types of notes in the
    # match snotes (only integers)
    divs_arg = [
        max(int((beat_type_map(note.OnsetInBeats) / 4)), 1)
        * note.Offset.denominator
        * (note.Offset.tuple_div or 1)
        for note in snotes
    ]
    divs_arg += [
        max(int((beat_type_map(note.OnsetInBeats) / 4)), 1)
        * note.Duration.denominator
        * (note.Duration.tuple_div or 1)
        for note in snotes
    ]

    onset_in_beats = np.array([note.OnsetInBeats for note in snotes])
    unique_onsets, inv_idxs = np.unique(onset_in_beats, return_inverse=True)
    # unique_onset_idxs = [np.where(onset_in_beats == u) for u in unique_onsets]

    iois_in_beats = np.diff(unique_onsets)
    beat_to_quarter = 4 / beat_type_map(onset_in_beats)

    iois_in_quarters_offset = np.r_[
        beat_to_quarter[0] * onset_in_beats[0],
        (4 / beat_type_map(unique_onsets[:-1])) * iois_in_beats,
    ]
    onset_in_quarters = np.cumsum(iois_in_quarters_offset)
    iois_in_quarters = np.diff(onset_in_quarters)

    # ___ these divs are relative to quarters;
    divs = np.lcm.reduce(np.unique(divs_arg))
    onset_in_divs = np.r_[0, np.cumsum(divs * iois_in_quarters)][inv_idxs]
    onset_in_quarters = onset_in_quarters[inv_idxs]

    # duration_in_beats = np.array([note.DurationInBeats for note in snotes])
    # duration_in_quarters = duration_in_beats * beat_to_quarter
    # duration_in_divs = duration_in_quarters * divs

    part.set_quarter_duration(0, divs)
    bars = np.unique([n.Measure for n in snotes])
    t = min_time
    t = t * 4 / beat_type_map(min_time)
    offset = t
    bar_times = {}

    if t > 0:
        # if we have an incomplete first measure that isn't an anacrusis
        # measure, add a rest (dummy)
        # t = t-t%beats_map(min_time)

        # if starting beat is above zero, add padding
        rest = score.Rest()
        part.add(rest, start=0, end=t * divs)
        onset_in_divs += t * divs
        offset = 0
        t = t - t % beats_map(min_time)

    for b0, b1 in iter_current_next(bars, end=bars[-1] + 1):

        bar_times.setdefault(b0, t)
        if t < 0:
            t = 0

        else:
            # multiply by diff between consecutive bar numbers
            n_bars = b1 - b0
            if t <= max_time_q:
                t += (n_bars * 4 * beats_map(t)) / beat_type_map(t)

    for ni, note in enumerate(snotes):
        # start of bar in quarter units
        bar_start = bar_times[note.Measure]

        on_off_scale = 1
        # on_off_scale = 1 means duration and beat offset are given in
        # whole notes, else they're given in beats (as in the KAIST data)
        if not match_offset_duration_in_whole:
            on_off_scale = beat_type_map(bar_start)

        # offset within bar in quarter units adjusted for different
        # time signatures -> 4 / beat_type_map(bar_start)
        bar_offset = (note.Beat - 1) * 4 / beat_type_map(bar_start)

        # offset within beat in quarter units adjusted for different
        # time signatures -> 4 / beat_type_map(bar_start)
        beat_offset = (
            4
            / on_off_scale
            * note.Offset.numerator
            / (note.Offset.denominator * (note.Offset.tuple_div or 1))
        )

        # check anacrusis measure beat counting type for the first note
        if bar_start < 0 and (bar_offset != 0 or beat_offset != 0) and ni == 0:
            # in case of fully counted anacrusis we set the bar_start
            # to -bar_duration (in quarters) so that the below calculation is correct
            # not active for shortened anacrusis measures
            bar_start = -beats_map(bar_start) * 4 / beat_type_map(bar_start)
            # reset the bar_start for other notes in the anacrusis measure
            bar_times[note.Bar] = bar_start

        # convert the onset time in quarters (0 at first barline) to onset
        # time in divs (0 at first note)
        onset_divs = int(round(divs * (bar_start + bar_offset + beat_offset - offset)))

        if not np.isclose(onset_divs, onset_in_divs[ni], atol=divs * 0.01):
            warnings.warn(
                "Calculated `onset_divs` does not match `OnsetInBeats` " "information!."
            )
            onset_divs = onset_in_divs[ni]
        assert onset_divs >= 0
        assert np.isclose(onset_divs, onset_in_divs[ni], atol=divs * 0.01)
        is_tied = False
        articulations = set()
        if "staccato" in note.ScoreAttributesList or "stac" in note.ScoreAttributesList:
            articulations.add("staccato")
        if "accent" in note.ScoreAttributesList:
            articulations.add("accent")
        if "leftOutTied" in note.ScoreAttributesList:
            is_tied = True

        # dictionary with keyword args with which the Note
        # (or GraceNote) will be instantiated
        note_attributes = dict(
            step=note.NoteName,
            octave=note.Octave,
            alter=note.Modifier,
            id=note.Anchor,
            articulations=articulations,
        )

        staff_nr = next(
            (a[-1] for a in note.ScoreAttributesList if a.startswith("staff")), None
        )
        try:
            note_attributes["staff"] = int(staff_nr)
        except (TypeError, ValueError):
            # no staff attribute, or staff attribute does not end with a number
            note_attributes["staff"] = None

        if "s" in note.ScoreAttributesList:
            note_attributes["voice"] = 1
        else:
            note_attributes["voice"] = next(
                (int(a) for a in note.ScoreAttributesList if number_pattern.match(a)),
                None,
            )

        # get rid of this if as soon as we have a way to iterate over the
        # duration components. For now we have to treat the cases simple
        # and compound durations separately.
        if note.Duration.add_components:
            prev_part_note = None

            for i, (num, den, tuple_div) in enumerate(note.Duration.add_components):
                # when we add multiple notes that are tied, the first note will
                # get the original note id, and subsequent notes will get a
                # derived note id (by appending, 'a', 'b', 'c',...)
                if i > 0:
                    # tnote_id = 'n{}_{}'.format(note.Anchor, i)
                    note_attributes["id"] = score._make_tied_note_id(
                        note_attributes["id"]
                    )

                part_note = score.Note(**note_attributes)

                # duration_divs from local beats --> 4/beat_type_map(bar_start)

                duration_divs = int(
                    (4 / on_off_scale) * divs * num / (den * (tuple_div or 1))
                )

                assert duration_divs > 0
                offset_divs = onset_divs + duration_divs
                part.add(part_note, onset_divs, offset_divs)

                if prev_part_note:
                    prev_part_note.tie_next = part_note
                    part_note.tie_prev = prev_part_note
                prev_part_note = part_note
                onset_divs = offset_divs

        else:
            num = note.Duration.numerator
            den = note.Duration.denominator
            tuple_div = note.Duration.tuple_div

            # duration_divs from local beats --> 4/beat_type_map(bar_start)
            duration_divs = int(
                divs * 4 / on_off_scale * num / (den * (tuple_div or 1))
            )
            offset_divs = onset_divs + duration_divs

            # notes with duration 0, are also treated as grace notes, even if
            # they do not have a 'grace' score attribute
            if "grace" in note.ScoreAttributesList or note.Duration.numerator == 0:
                part_note = score.GraceNote(
                    grace_type="appoggiatura", **note_attributes
                )

            else:
                part_note = score.Note(**note_attributes)

            part.add(part_note, onset_divs, offset_divs)

        # Check if the note is tied and if so, add the tie information
        if is_tied:
            found = False
            # iterate over all notes in the Timeline that end at the starting point.
            for el in part_note.start.iter_ending(score.Note):
                if isinstance(el, score.Note):
                    condition = el.step == note_attributes["step"] and el.octave == note_attributes["octave"] and el.alter == note_attributes["alter"]
                    if condition:
                        el.tie_next = part_note
                        part_note.tie_prev = el
                        found = True
                        break
            if not found:
                warnings.warn(
                    "Tie information found, but no previous note found to tie to for note {}.".format(part_note.id)
                )

    # add time signatures
    for (ts_beat_time, ts_bar, tsg) in ts:
        ts_beats = tsg.numerator
        ts_beat_type = tsg.denominator
        # check if time signature is in a known measure (from notes)
        if ts_bar in bar_times.keys():
            bar_start_divs = int(divs * (bar_times[ts_bar] - offset))  # in quarters
            bar_start_divs = max(0, bar_start_divs)
        else:
            bar_start_divs = 0
        part.add(score.TimeSignature(ts_beats, ts_beat_type), bar_start_divs)

    # add key signatures
    for (ks_beat_time, ks_bar, keys) in mf.key_signatures:

        if ks_bar in bar_times.keys():
            bar_start_divs = int(divs * (bar_times[ks_bar] - offset))  # in quarters
            bar_start_divs = max(0, bar_start_divs)
        else:
            bar_start_divs = 0

        # TODO
        # * use key estimation if there are multiple defined keys
        # fifths, mode = key_name_to_fifths_mode(key_name)
        part.add(score.KeySignature(keys.fifths, keys.mode), ks_bar)

    add_staffs(part)
    # add_clefs(part)

    # add incomplete measure if necessary
    if offset < 0:
        part.add(score.Measure(number=0), 0, int(-offset * divs))

    # add the rest of the measures automatically
    score.add_measures(part)
    score.tie_notes(part)
    score.find_tuplets(part)

    if not all([n.voice for n in part.notes_tied]):
        for note in part.notes_tied:
            if note.voice is None:
                note.voice = 1

    return part


def make_timesig_maps(
    ts_orig: List[Tuple[float, int, MatchTimeSignature]],
    max_time: float,
) -> (Callable, Callable, Callable, Callable, float, float):
    """
    Create time signature (interpolation) maps

    Parameters
    ----------
    ts_orig : List[Tuple[float, int, MatchTimeSignature]]
        A list of tuples containing position in beats, measure and
        MatchTimeSignature instances
    max_time : float
        Maximal time of the time signatures

    Returns
    -------
    beats_map: callable
    qbeats_map: callable
    beat_type_map: callable
    qbeat_type_map: callable
    start_q: float
    end_q: float
    """
    # TODO: make sure that ts_orig covers range from min_time
    # return two functions that map score times (in quarter units) to time sig
    # beats, and time sig beat_type respectively
    ts = list(ts_orig)
    assert len(ts) > 0
    ts.append((max_time, None, ts[-1][2]))

    x = np.array([t for t, _, _ in ts])
    y = np.array([(x.numerator, x.denominator) for _, _, x in ts])

    start_q = x[0] * 4 / y[0, 1]
    x_q = np.cumsum(np.r_[start_q, 4 * np.diff(x) / y[:-1, 1]])
    end_q = x_q[-1]

    # TODO: fix error with bounds
    qbeats_map = interp1d(
        x_q,
        y[:, 0],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 0], y[-1, 0]),
    )
    qbeat_type_map = interp1d(
        x_q,
        y[:, 1],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 1], y[-1, 1]),
    )
    beats_map = interp1d(
        x,
        y[:, 0],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 0], y[-1, 0]),
    )
    beat_type_map = interp1d(
        x,
        y[:, 1],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 1], y[-1, 1]),
    )

    return beats_map, qbeats_map, beat_type_map, qbeat_type_map, start_q, end_q


def add_staffs(part: Part, split: int = 55, only_missing: bool = True) -> None:
    """
        Method to add staff information to a part

        Parameters
        ----------
        part: Part
            Part to add staff information to.
        split: int
            MIDI pitch to split staff into upper and lower. Default is 55
        only_missing: bool
            If True, only add staff to those notes that do not have staff info already.
    x"""
    # assign staffs using a hard limit
    notes = part.notes_tied
    for n in notes:

        if only_missing and n.staff:
            continue

        if n.midi_pitch > split:
            staff = 1
        else:
            staff = 2

        n.staff = staff

        n_tied = n.tie_next
        while n_tied:
            n_tied.staff = staff
            n_tied = n_tied.tie_next

    part.add(score.Clef(staff=1, sign="G", line=2, octave_change=0), 0)
    part.add(score.Clef(staff=2, sign="F", line=4, octave_change=0), 0)


if __name__ == "__main__":

    pass
