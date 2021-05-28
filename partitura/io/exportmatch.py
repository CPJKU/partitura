#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for exporting matchfiles
"""
import logging
import numpy as np

from scipy.interpolate import interp1d

from partitura.io.importmatch import (
    MatchInfo,
    MatchMeta,
    MatchSnote,
    MatchNote,
    MatchSnoteNote,
    MatchSnoteDeletion,
    MatchInsertionNote,
    MatchSustainPedal,
    MatchSoftPedal,
    MatchOrnamentNote,
    MatchTrillNote,
    MatchFile,
)
import partitura.score as score
from partitura.utils.music import midi_pitch_to_pitch_spelling, MAJOR_KEYS, MINOR_KEYS

__all__ = ["save_match"]

LOGGER = logging.getLogger(__name__)


def seconds_to_midi_ticks(t, mpq=500000, ppq=480):
    return int(np.round(10 ** 6 * ppq * t / mpq))


def _fifths_mode_to_match_key_name(fifths, mode):
    if mode == "minor":
        keylist = MINOR_KEYS
        suffix = "min"
    elif mode == "major":
        keylist = MAJOR_KEYS
        suffix = "Maj"
    else:
        raise Exception("Unknown mode {0}".format(mode))

    try:
        name = keylist[fifths + 7]
    except IndexError:
        raise Exception("Unknown number of fifths {}".format(fifths))

    return "{0} {1}".format(name, suffix)


def fifths_mode_to_match_key_name(fifths, mode=None):
    if mode is None:
        key_sig = "{0}/{1}".format(
            _fifths_mode_to_match_key_name(fifths, "major"),
            _fifths_mode_to_match_key_name(fifths, "minor"),
        )
    else:
        key_sig = _fifths_mode_to_match_key_name(fifths, mode)

    return key_sig


def matchfile_from_alignment(
    alignment,
    ppart,
    spart,
    mpq=500000,
    ppq=480,
    performer=None,
    composer=None,
    piece=None,
    magaloff_zeilinger_quirk=False,
):
    """
    Generate a MatchFile object from an Alignment, a PerformedPart and
    a Part

    Parameters
    ----------
    alignment : list
        A list of dictionaries containing alignment information.
        See `partitura.io.importmatch.alignment_from_matchfile`.
    ppart : partitura.performance.PerformedPart
        An instance of `PerformedPart` containing performance information.
    spart : partitura.score.Part
        An instance of `Part` containing score information.
    mpq : int
        Milliseconds per quarter note.
    ppq: int
        Parts per quarter note.
    performer : str or None
        Name(s) of the performer(s) of the `PerformedPart`.
    composer : str or None
        Name(s) of the composer(s) of the piece represented by `Part`.
    piece : str or None:
        Name of the piece represented by `Part`.

    Returns
    -------
    matchfile : MatchFile
        An instance of `partitura.io.importmatch.MatchFile`.
    """
    # Information for the header
    header_lines = dict()
    header_lines["version"] = MatchInfo(Attribute="matchFileVersion", Value="5.0")
    if performer is not None:
        header_lines["performer"] = MatchInfo(Attribute="performer", Value=performer)
    if piece is not None:
        header_lines["piece"] = MatchInfo(Attribute="piece", Value=piece)
    if composer is not None:
        header_lines["composer"] = MatchInfo(Attribute="composer", Value=composer)

    header_lines["clock_units"] = MatchInfo(Attribute="midiClockUnits", Value=ppq)
    header_lines["clock_rate"] = MatchInfo(Attribute="midiClockRate", Value=mpq)

    # Measure map (which measure corresponds to each timepoint
    measure_starts = np.array(
        [(m.number, m.start.t) for m in spart.iter_all(score.Measure)]
    )
    measure_map = interp1d(
        measure_starts[:, 1],
        measure_starts[:, 0],
        kind="previous",
        bounds_error=False,
        fill_value=(measure_starts[:, 0].min(), measure_starts[:, 0].max()),
    )

    # Create MatchSnotes from score information
    score_info = dict()
    for i, m in enumerate(spart.iter_all(score.Measure)):

        # Get all notes in the measure (bar is the terminology
        # used in the definition of MatchFiles)
        snotes = spart.iter_all(score.Note, m.start, m.end, include_subclasses=True)
        # Beginning of each measure

        # ____ does this really give the full measure? or just the first note?
        # it seems like it returns the first note.
        bar_start = float(spart.beat_map(m.start.t))

        for n in snotes:
            # Get note information
            # TODO: preserve symbolic durations?
            # TODO: correct beat and moffset calculation
            bar = int(m.number)
            onset, offset = spart.beat_map([n.start.t, n.start.t + n.duration_tied])
            duration = offset - onset
            beat = (onset - bar_start) // 1
            ts_num, ts_den = spart.time_signature_map(n.start.t)
            # In metrical offset in whole notes
            moffset = (onset - bar_start - beat) / ts_den
            # offset = onset + duration
            # print("DURATION", duration, n.duration_tied)
            score_info[n.id] = MatchSnote(
                Anchor=n.id,
                NoteName=n.step,
                Modifier=n.alter if n.alter is not None else 0,
                Octave=n.octave,
                Bar=int(bar),
                Beat=int(beat) + 1,
                Offset=float(moffset),
                Duration=float(duration) / ts_den,
                OnsetInBeats=float(onset),
                OffsetInBeats=float(offset),
            )

    # Create MatchNotes from performance informaton
    perf_info = dict()

    for pn in ppart.notes:
        note_name, modifier, octave = midi_pitch_to_pitch_spelling(pn["midi_pitch"])
        onset = seconds_to_midi_ticks(pn["note_on"], mpq=mpq, ppq=ppq)
        offset = seconds_to_midi_ticks(pn["note_off"], mpq=mpq, ppq=ppq)
        adjoffset = seconds_to_midi_ticks(pn["sound_off"], mpq=mpq, ppq=ppq)
        perf_info[pn["id"]] = MatchNote(
            Number=pn["id"],
            NoteName=note_name,
            Modifier=modifier,
            Octave=octave,
            Onset=onset,
            Offset=offset,
            AdjOffset=adjoffset,
            Velocity=pn["velocity"],
            MidiPitch=pn["midi_pitch"],
        )

    # Create match lines for note information
    note_lines = []
    for al_note in alignment:

        label = al_note["label"]

        # Create match line for matched score and performed notes
        if label == "match":
            # quirk from Magaloff/Zeilinger
            if magaloff_zeilinger_quirk:
                al_note["score_id"] = al_note["score_id"].split("-")[0]
            snote = score_info[al_note["score_id"]]

            pnote = perf_info[al_note["performance_id"]]
            snote_note_line = MatchSnoteNote(snote=snote, note=pnote)
            note_lines.append(snote_note_line)

        # Matchline for deleted notes
        elif label == "deletion":
            # Quirk for Magaloff/Zeilinger
            if magaloff_zeilinger_quirk:
                al_note["score_id"] = al_note["score_id"].split("-")[0]
            snote = score_info[al_note["score_id"]]
            deletion_line = MatchSnoteDeletion(snote)
            note_lines.append(deletion_line)

        # Matchline for inserted notes
        elif label == "insertion":
            note = perf_info[al_note["performance_id"]]
            insertion_line = MatchInsertionNote(note)
            note_lines.append(insertion_line)

        elif label == "ornament":
            ornament_type = al_note["type"]
            # Quirk for Magaloff/Zeilinger
            if magaloff_zeilinger_quirk:
                al_note["score_id"] = al_note["score_id"].split("-")[0]
                # al_note['score_id'] = 'n' + al_note['score_id'].split('-')[0]
            snote = score_info[al_note["score_id"]]
            note = perf_info[al_note["performance_id"]]
            if ornament_type == "trill":
                ornament_line = MatchTrillNote(Anchor=snote.Anchor, note=note)
            else:
                print(ornament_type)
                ornament_line = MatchOrnamentNote(Anchor=snote.Anchor, note=note)

            note_lines.append(ornament_line)

        else:
            print("unprocessed line {0}".format(label))
            print(str(al_note))

    # Create match lines for pedal information
    pedal_lines = []
    for c in ppart.controls:
        t = seconds_to_midi_ticks(c["time"], mpq=mpq, ppq=ppq)
        value = int(c["value"])
        if c["number"] == 64:  # c['type'] == 'sustain_pedal':
            sustain_pedal = MatchSustainPedal(Time=t, Value=value)
            pedal_lines.append(sustain_pedal)

        if c["number"] == 67:  # c['type'] == 'soft_pedal':
            soft_pedal = MatchSoftPedal(Time=t, Value=value)
            pedal_lines.append(soft_pedal)

    # Create match lines for meta information
    meta_lines = []
    for i, ts in enumerate(spart.iter_all(score.TimeSignature)):
        value = "{0}/{1}".format(int(ts.beats), int(ts.beat_type))
        if i == 0:
            first_ts = MatchInfo(Attribute="timeSignature", Value="[{0}]".format(value))
            header_lines["time_signature"] = first_ts

        ts_line = MatchMeta(
            Attribute="timeSignature",
            Value=value,
            Bar=int(measure_map(ts.start.t)),
            TimeInBeats=float(spart.beat_map(ts.start.t)),
        )

        meta_lines.append(ts_line)

    for i, ks in enumerate(spart.iter_all(score.KeySignature)):
        value = fifths_mode_to_match_key_name(ks.fifths, ks.mode)
        if i == 0:
            first_ks = MatchInfo(Attribute="keySignature", Value="[{0}]".format(value))
            header_lines["key_signature"] = first_ks

        ks_line = MatchMeta(
            Attribute="keySignature",
            Value=value,
            Bar=int(measure_map(ks.start.t)),
            TimeInBeats=float(spart.beat_map(ks.start.t)),
        )
        meta_lines.append(ks_line)

    # Construct header of match file
    header_order = [
        "version",
        "piece",
        "composer",
        "performer",
        "clock_units",
        "clock_rate",
        "key_signature",
        "time_signature",
    ]
    all_match_lines = []
    for h in header_order:
        if h in header_lines:
            all_match_lines.append(header_lines[h])

    # Concatenate all lines
    all_match_lines += note_lines + meta_lines + pedal_lines

    # Create match file
    matchfile = MatchFile.from_lines(lines=all_match_lines, name=ppart.part_name)

    return matchfile


def save_match(
    alignment,
    ppart,
    spart,
    out,
    mpq=500000,
    ppq=480,
    performer=None,
    composer=None,
    piece=None,
):
    """
    Save an Alignment of a PerformedPart to a Part in a match file.

    Parameters
    ----------
    alignment : list
        A list of dictionaries containing alignment information.
        See `partitura.io.importmatch.alignment_from_matchfile`.
    ppart : partitura.performance.PerformedPart
        An instance of `PerformedPart` containing performance information.
    spart : partitura.score.Part
        An instance of `Part` containing score information.
    out : str
        Out to export the matchfile.
    mpq : int
        Milliseconds per quarter note.
    ppq: int
        Parts per quarter note.
    performer : str or None
        Name(s) of the performer(s) of the `PerformedPart`.
    composer : str or None
        Name(s) of the composer(s) of the piece represented by `Part`.
    piece : str or None:
        Name of the piece represented by `Part`.
    """
    # Get matchfile
    matchfile = matchfile_from_alignment(
        alignment=alignment,
        ppart=ppart,
        spart=spart,
        mpq=mpq,
        ppq=ppq,
        performer=performer,
        composer=composer,
        piece=piece,
    )
    # write matchfile
    matchfile.write(out)
