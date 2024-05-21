#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods for synthesizing score- or performance-like
objects using fluidsynth. Fluidsynth is an optional dependency.
"""

import os
from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import partitura as pt

try:
    from fluidsynth import Synth

    HAS_FLUIDSYNTH = True
except ImportError:  # pragma: no cover
    Synth = None  # pragma: no cover
    HAS_FLUIDSYNTH = False  # pragma: no cover

from partitura.utils.synth import SAMPLE_RATE
from partitura.performance import PerformanceLike
from partitura.score import ScoreLike
from partitura.utils.misc import PathLike, download_file
from partitura.utils.music import (
    ensure_notearray,
    get_time_units_from_note_array,
    performance_notearray_from_score_notearray,
)

# MuseScore's soundfont distributed under the MIT License.
# https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General_License.md
DEFAULT_SOUNDFONT_URL = "ftp://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3"

DEFAULT_SOUNDFONT = os.path.join(
    pt.__path__[0],
    "assets",
    "MuseScore_General.sf3",
)

if not os.path.exists(DEFAULT_SOUNDFONT) and HAS_FLUIDSYNTH:  # pragma: no cover
    print(f"Downloading soundfont from {DEFAULT_SOUNDFONT_URL}...")  # pragma: no cover
    download_file(
        url=DEFAULT_SOUNDFONT_URL,
        out=DEFAULT_SOUNDFONT,
    )  # pragma: no cover


def synthesize_fluidsynth(
    note_info: Union[ScoreLike, PerformanceLike, np.ndarray],
    samplerate: int = SAMPLE_RATE,
    soundfont: PathLike = DEFAULT_SOUNDFONT,
    bpm: Union[float, np.ndarray, Callable] = 60,
) -> np.ndarray:
    """
    Synthesize partitura object with note information using
    fluidsynth.

    Parameters
    ----------
    note_info : ScoreLike, PerformanceLike or np.ndarray
        A partitura object with note information.

    samplerate: int
        The sample rate of the audio file in Hz.

    soundfont: PathLike
        The path to the soundfont (in SF2/SF3 format).

    bpm : float, np.ndarray or callable
        The bpm to render the output (if the input is a score-like object).
        See `partitura.utils.music.performance_notearray_from_score_notearray`
        for more information on this parameter.

    Returns
    -------
    output_audio_signal : np.ndarray
       Audio signal as a 1D array.
    """

    if not HAS_FLUIDSYNTH:
        raise ImportError("Fluidsynth is not installed!")  # pragma: no cover

    if isinstance(note_info, pt.performance.Performance):
        for ppart in note_info:
            ppart.sustain_pedal_threshold = 127

    if isinstance(note_info, pt.performance.PerformedPart):
        note_info.sustain_pedal_threshold = 127
    note_array = ensure_notearray(note_info)

    onset_unit, _ = get_time_units_from_note_array(note_array)
    if np.min(note_array[onset_unit]) <= 0:
        note_array[onset_unit] = note_array[onset_unit] + np.min(note_array[onset_unit])

    pitch = note_array["pitch"]
    # If the input is a score, convert score time to seconds
    if onset_unit != "onset_sec":
        pnote_array = performance_notearray_from_score_notearray(
            snote_array=note_array,
            bpm=bpm,
        )
        onsets = pnote_array["onset_sec"]
        offsets = pnote_array["onset_sec"] + pnote_array["duration_sec"]
        # duration = pnote_array["duration_sec"]
        channel = pnote_array["channel"]
        track = pnote_array["track"]
        velocity = pnote_array["velocity"]
    else:
        onsets = note_array["onset_sec"]
        offsets = note_array["onset_sec"] + note_array["duration_sec"]

        if "velocity" in note_array.dtype.names:
            velocity = note_array["velocity"]
        else:
            velocity = np.ones(len(onsets), dtype=int) * 64
        if "channel" in note_array.dtype.names:
            channel = note_array["channel"]
        else:
            channel = np.zeros(len(onsets), dtype=int)

        if "track" in note_array.dtype.names:
            track = note_array["track"]
        else:
            track = np.zeros(len(onsets), dtype=int)

    controls = []
    if isinstance(note_info, pt.performance.Performance):

        for ppart in note_info:
            controls += ppart.controls

    unique_tracks = list(
        set(list(np.unique(track)) + list(set([c["track"] for c in controls])))
    )

    track_dict = defaultdict(lambda: defaultdict(list))

    for tn in unique_tracks:
        track_idxs = np.where(track == tn)[0]

        track_channels = channel[track_idxs]
        track_pitch = pitch[track_idxs]
        track_onsets = onsets[track_idxs]
        track_offsets = offsets[track_idxs]
        track_velocity = velocity[track_idxs]

        unique_channels = np.unique(track_channels)

        track_controls = [c for c in controls if c["track"] == tn]

        for chn in unique_channels:

            channel_idxs = np.where(track_channels == chn)[0]

            channel_pitch = track_pitch[channel_idxs]
            channel_onset = track_onsets[channel_idxs]
            channel_offset = track_offsets[channel_idxs]
            channel_velocity = track_velocity[channel_idxs]

            channel_controls = [c for c in track_controls if c["channel"] == chn]

            track_dict[tn][chn] = [
                channel_pitch,
                channel_onset,
                channel_offset,
                channel_velocity,
                channel_controls,
            ]

    # set to mono
    synthesizer = Synth(samplerate=SAMPLE_RATE)
    sf_id = synthesizer.sfload(soundfont)

    audio_signals = []
    for tn, channel_info in track_dict.items():

        for chn, (pi, on, off, vel, ctrls) in channel_info.items():

            audio_signal = synth_note_info(
                pitch=pi,
                onsets=on,
                offsets=off,
                velocities=vel,
                controls=ctrls,
                program=None,
                synthesizer=synthesizer,
                sf_id=sf_id,
                channel=chn,
                samplerate=samplerate,
            )
            audio_signals.append(audio_signal)

    # pad audio signals:

    signal_lengths = [len(signal) for signal in audio_signals]
    max_len = np.max(signal_lengths)

    output_audio_signal = np.zeros(max_len)

    for sl, audio_signal in zip(signal_lengths, audio_signals):

        output_audio_signal[:sl] += audio_signal

    # normalization term
    norm_term = max(audio_signal.max(), abs(audio_signal.min()))
    output_audio_signal /= norm_term

    return output_audio_signal


def synth_note_info(
    pitch: np.ndarray,
    onsets: np.ndarray,
    offsets: np.ndarray,
    velocities: np.ndarray,
    controls: Optional[list],
    program: Optional[int],
    synthesizer: Synth,
    sf_id: int,
    channel: int,
    samplerate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Synthesize note information with Fluidsynth.
    This method is designed to synthesize the notes in a
    single track and channel.

    Parameters
    ----------
    pitch : np.ndarray
        An array with pitch information for each note.
    onsets : np.ndarray
        An array with onset time in seconds for each note.
    offsets : np.ndarray
        An array with offset times in seconds for each note.
    velocities : np.ndarray
        An array with MIDI velocities for each note.
    controls : Optional[list]
        A list of MIDI controls (e.g., pedals).
        (as the `controls` attribute in `PerformedPart` objects)
    program : Optional[int]
        A list of MIDI programs as dictionaries
        (as the `program` attribute in `PerformedPart` objects).
    synthesizer : Synth
        An instance of a fluidsynth Synth object.
    sf_id : int
        The id of the synthesizer object
    channel : int
        Channel for the the notes.
    samplerate : int, optional
        Sample rate, by default SAMPLE_RATE

    Returns
    -------
    audio_signal : np.ndarray
        A 1D array with the synthesized audio signal.
    """

    # set program
    synthesizer.program_select(channel, sf_id, 0, program or 0)

    if len(controls) > 0 and len(offsets) > 0:
        piece_duration = max(offsets.max(), np.max([c["time"] for c in controls]))
    elif len(controls) > 0 and len(offsets) == 0:
        piece_duration = np.max([c["time"] for c in controls])
    elif len(controls) == 0 and len(offsets) > 0:
        piece_duration = offsets.max()
    else:
        # return a single zero
        audio_signal = np.zeros(1)
        return audio_signal

    num_frames = int(np.round(piece_duration * samplerate))

    # Initialize array containing audio
    audio_signal = np.zeros(num_frames, dtype="float")

    # Initialize the time axis
    x = np.linspace(0, piece_duration, num=num_frames)

    # onsets in frames (i.e., indices of the `audio_signal` array)
    onsets_in_frames = np.searchsorted(x, onsets, side="left")
    offsets_in_frames = np.searchsorted(x, offsets, side="left")

    messages = []
    for ctrl in controls or []:

        messages.append(
            (
                "cc",
                channel,
                ctrl["number"],
                ctrl["value"],
                np.searchsorted(x, ctrl["time"], side="left"),
            )
        )

    for pi, vel, oif, ofif in zip(
        pitch, velocities, onsets_in_frames, offsets_in_frames
    ):

        messages += [
            ("noteon", channel, pi, vel, oif),
            ("noteoff", channel, pi, ofif),
        ]

    # sort messages
    messages.sort(key=lambda x: x[-1])

    delta_times = [
        int(nm[-1] - cm[-1]) for nm, cm in zip(messages[1:], messages[:-1])
    ] + [0]

    for dt, msg in zip(delta_times, messages):

        msg_type = msg[0]
        msg_time = msg[-1]
        getattr(synthesizer, msg_type)(*msg[1:-1])

        samples = synthesizer.get_samples(dt)[::2]
        audio_signal[msg_time : msg_time + dt] = samples

    return audio_signal
