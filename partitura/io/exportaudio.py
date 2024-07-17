#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods to synthesize a Partitura ScoreLike object to wav.
"""
from typing import Union, Optional, Callable, Dict, Any
import numpy as np
from scipy.io import wavfile


from partitura.score import ScoreLike
from partitura.performance import PerformanceLike

from partitura.utils.synth import synthesize, SAMPLE_RATE, A4
from partitura.utils.fluidsynth import (
    synthesize_fluidsynth,
    DEFAULT_SOUNDFONT,
    HAS_FLUIDSYNTH,
)

from partitura.utils.misc import PathLike

__all__ = [
    "save_wav",
    "save_wav_fluidsynth",
]


def save_wav(
    input_data: Union[ScoreLike, PerformanceLike, np.ndarray],
    out: Optional[PathLike] = None,
    samplerate: int = SAMPLE_RATE,
    envelope_fun: Union[str, Callable] = "linear",
    tuning: Union[str, Callable] = "equal_temperament",
    tuning_kwargs: Dict[str, Any] = {"a4": A4},
    harmonic_dist: Optional[Union[str, int]] = None,
    bpm: Union[float, np.ndarray, Callable] = 60,
) -> Optional[np.ndarray]:
    """
    Export a score (a `Score`, `Part`, `PartGroup` or list of `Part` instances),
    a performance (`Performance`, `PerformedPart` or list of `PerformedPart` instances)
    as a WAV file using additive synthesis


    Parameters
    ----------
    input_data : ScoreLike, PerformanceLike or np.ndarray
        A partitura object with note information.
    out : PathLike or None
        Path of the output Wave file. If None, the method outputs
        the audio signal as an array (see `audio_signal` below).
    samplerate: int
        The sample rate of the audio file in Hz. The default is 44100Hz.
    envelope_fun: {"linear", "exp" } or callable
        The type of envelop to apply to the individual sine waves.
        If "linear" or "exp", the methods `lin_in_lin_out` and `exp_in_exp_out`
        in `partitura.utils.synth` will be used. Otherwise this argument should
        be a callable. See `lin_in_lin_out` for more details.
    tuning: {"equal_temperament", "natural"} or callable.
        The tuning system to use. If the value is "equal_temperament",
        12 tone equal temperament implemented in
        `partitura.utils.music.midi_pitch_to_frequency` will be used. If the value is
        "natural", the function `partitura.utils.synth.midi_pitch_to_tempered_frequency`
        will be used. Note that `midi_pitch_to_tempered_frequency` computes
        intervals (and thus, frequencies) with respect to a reference note
        (A4 by default) and uses the interval ratios specified by
        `partitura.utils.synth.FIVE_LIMIT_INTERVAL_RATIOS`. See the documentation of
        these functions for more information. If a callable is provided, function should
        get MIDI pitch as input and return frequency in Hz as output.
    tuning_kwargs : dict
        Dictionary of keyword arguments to be passed to the tuning function
        specified in  `tuning`. See `midi_pitch_to_tempered_frequency` and
       `midi_pitch_to_frequency` for more information on their keyword arguments.
    harmonic_dist : int,  "shepard" or None (optional)
        Distribution of harmonics. If an integer, it is the number
        of harmonics to be considered. If "shepard", it uses Shepard tones.
        Default is None (i.e., only consider the fundamental frequency)
    bpm : float, np.ndarray, callable
        The bpm to render the output (if the input is a score-like object).
        See `partitura.utils.music.performance_notearray_from_score_notearray`
        for more information on this parameter.

    Returns
    -------
    audio_signal : np.ndarray
       Audio signal as a 1D array. Only returned if `out` is None.
    """
    # synthesize audio signal
    audio_signal = synthesize(
        note_info=input_data,
        samplerate=samplerate,
        envelope_fun=envelope_fun,
        tuning=tuning,
        tuning_kwargs=tuning_kwargs,
        harmonic_dist=harmonic_dist,
        bpm=bpm,
    )

    if out is not None:
        # convert to 16bit integers (save as PCM 16 bit)
        # (some DAWs cannot load audio files that are float64,
        # e.g., Logic)
        amplitude = np.iinfo(np.int16).max
        if abs(audio_signal).max() <= 1:
            # convert to 16bit integers (save as PCM 16 bit)
            amplitude = np.iinfo(np.int16).max
            audio_signal *= amplitude
        wavfile.write(out, samplerate, audio_signal.astype(np.int16))

    else:
        return audio_signal


def save_wav_fluidsynth(
    input_data: Union[ScoreLike, PerformanceLike, np.ndarray],
    out: Optional[PathLike] = None,
    samplerate: int = SAMPLE_RATE,
    soundfont: PathLike = DEFAULT_SOUNDFONT,
    bpm: Union[float, np.ndarray, Callable] = 60,
) -> Optional[np.ndarray]:
    """
    Export a score (a `Score`, `Part`, `PartGroup` or list of `Part` instances),
    a performance (`Performance`, `PerformedPart` or list of `PerformedPart` instances)
    as a WAV file using fluidsynth

    Parameters
    ----------
    input_data : ScoreLike, PerformanceLike or np.ndarray
        A partitura object with note information.
    out : PathLike or None
        Path of the output Wave file. If None, the method outputs
        the audio signal as an array (see `audio_signal` below).
    samplerate: int
        The sample rate of the audio file in Hz. The default is 44100Hz.
    soundfont : PathLike
        Path to the soundfont in SF2/SF3 format for fluidsynth.
    bpm : float, np.ndarray, callable
        The bpm to render the output (if the input is a score-like object).
        See `partitura.utils.music.performance_notearray_from_score_notearray`
        for more information on this parameter.

    Returns
    -------
    audio_signal : np.ndarray
       Audio signal as a 1D array. Only returned if `out` is None.
    """

    if not HAS_FLUIDSYNTH:
        raise ImportError("Fluidsynth is not installed!")

    audio_signal = synthesize_fluidsynth(
        note_info=input_data,
        samplerate=samplerate,
        soundfont=soundfont,
        bpm=bpm,
    )

    if out is not None:
        # Write audio signal

        # convert to 16bit integers (save as PCM 16 bit)
        # (some DAWs cannot load audio files that are float64,
        # e.g., Logic)
        amplitude = np.iinfo(np.int16).max
        if abs(audio_signal).max() <= 1:
            # convert to 16bit integers (save as PCM 16 bit)
            amplitude = np.iinfo(np.int16).max
            audio_signal *= amplitude
        wavfile.write(out, samplerate, audio_signal.astype(np.int16))
    else:
        return audio_signal
