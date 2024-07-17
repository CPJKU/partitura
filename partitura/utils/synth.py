#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods for synthesizing score- or performance-like
objects using additive synthesis

TODO
----
* Add other tuning systems?
"""
from __future__ import annotations
from typing import Union, Tuple, Dict, Optional, Any, Callable, TYPE_CHECKING

import numpy as np

from scipy.interpolate import interp1d


from partitura.utils.music import (
    A4,
    ensure_notearray,
    get_time_units_from_note_array,
    midi_pitch_to_frequency,
    performance_notearray_from_score_notearray,
)
from partitura.utils.globals import (
    DTYPE,
    SAMPLE_RATE,
    TWO_PI,
    FIVE_LIMIT_INTERVAL_RATIOS,
    A4,
    NATURAL_INTERVAL_RATIOS,
)

if TYPE_CHECKING:
    # Import typing info for typing annotations.
    # For this to work we need to import annotations from __future__
    # Solution from
    # https://medium.com/quick-code/python-type-hinting-eliminating-importerror-due-to-circular-imports-265dfb0580f8
    from partitura.score import ScoreLike, Interval
    from partitura.performance import PerformanceLike, Performance, PerformedPart


def midi_pitch_to_natural_frequency(
    midi_pitch: Union[int, float, np.ndarray],
    a4: Union[int, float] = A4,
    natural_interval_ratios: Dict[int, float] = NATURAL_INTERVAL_RATIOS,
) -> Union[float, np.ndarray]:
    """
    Convert MIDI pitch to frequency in Hz using natural tunning (i.e., with
    respect to the harmonic series).
    This method computes intervals with respect to A4.

    Parameters
    ----------
    midi_pitch: int, float or ndarray
        MIDI pitch of the note(s).
    a4 : int or float (optional)
        Frequency of A4 in Hz. By default is 440 Hz.

    Returns
    -------
    freq : float or ndarray
        Frequency of the note(s).

    Notes
    -----
    This implementation computes the natural interval ratios
    (with respect to the harmonic series), but with respect to
    octaves centered on A. All intervals are computed with respect
    to the A in the same octave as the note in question (e.g.,
    C4 is a descending major sixth with respect to A4, E5 is descending
    perfect fourth computed with respect to A5, etc.).

    """

    octave = (midi_pitch // 12) - 1

    aref = 69.0 - 12.0 * (4 - octave)

    aref_freq = a4 / (2.0 ** ((4 - octave)))

    interval = midi_pitch - aref

    if isinstance(interval, (int, float)):
        interval = np.array([interval], dtype=int)

    ratios = np.array(
        [
            natural_interval_ratios[abs(itv)] ** (1 if itv >= 0 else -1)
            for itv in interval
        ]
    )

    freqs = aref_freq * ratios

    if isinstance(midi_pitch, (int, float)):
        freqs = float(freqs)
    return freqs


def midi_pitch_to_tempered_frequency(
    midi_pitch: Union[int, float, np.ndarray],
    reference_midi_pitch: Union[int, float] = 69,
    reference_frequency: float = A4,
    interval_ratios: Dict[int, float] = FIVE_LIMIT_INTERVAL_RATIOS,
) -> Union[float, np.ndarray]:
    """
    Convert MIDI pitch to frequency in Hz using
    a temperament given as frequency ratios above
    a reference pitch.

    Parameters
    ----------
    midi_pitch: int, float or ndarray
        MIDI pitch of the note(s).
    reference_midi_pitch : int or float (optional)
        midi pitch of the reference pitch. By default is 69 (A4).
    reference_frequency : int (optional)
        Frequency of A4 in Hz. By default is 440 Hz.
    interval_ratios: dict
        Dictionary of interval ratios from the reference

    Returns
    -------
    freq : float or ndarray
        Frequency of the note(s).
    """

    interval = (midi_pitch - reference_midi_pitch) % 12
    octave = (midi_pitch - reference_midi_pitch) // 12
    adjusted_reference_frequency = reference_frequency / (2.0**-octave)

    if isinstance(interval, (int, float)):
        interval = np.array([interval], dtype=int)

    ratios = np.array([interval_ratios[abs(itv)] for itv in interval])

    freqs = adjusted_reference_frequency * ratios

    if isinstance(midi_pitch, (int, float)):
        freqs = float(freqs)
    return freqs


def exp_in_exp_out(
    num_frames: int,
    dtype: type = DTYPE,
) -> np.ndarray:
    """
    Sound envelope with exponential attack and decay

    Parameters
    ----------
    num_frames : int
        Size of the window in frames.

    Returns
    -------
    envelope : np.ndarray
        1D array with the envelope.
    """
    # Initialize envelope
    envelope = np.ones(num_frames, dtype=dtype)
    # number of frames for decay
    decay_frames = np.minimum(num_frames // 10, 1000)
    # number of frames for attack
    attack_frames = np.minimum(num_frames // 100, 1000)
    # Compute envelope
    envelope[-decay_frames:] = np.exp(-np.linspace(0, 100, decay_frames)).astype(dtype)
    envelope[:attack_frames] = np.exp(np.linspace(-100, 0, attack_frames)).astype(dtype)

    return envelope


def lin_in_lin_out(num_frames: int, dtype: type = DTYPE) -> np.ndarray:
    """
    Sound envelope with linear attack and decay

    Parameters
    ----------
    num_frames : int
        Size of the window in frames.

    Returns
    -------
    envelope : np.ndarray
        1D array with the envelope.
    """
    # Initialize envelope
    envelope = np.ones(num_frames, dtype=dtype)
    # Number of frames for decay
    decay_frames = np.minimum(num_frames // 10, 1000)
    # number of frames for attack
    attack_frames = np.minimum(num_frames // 100, 1000)
    # Compute envelope
    envelope[-decay_frames:] = np.linspace(1, 0, decay_frames, dtype=dtype)
    envelope[:attack_frames] = np.linspace(0, 1, attack_frames, dtype=dtype)
    return envelope


def additive_synthesis(
    freqs: Union[int, float, np.ndarray],
    duration: float,
    samplerate: Union[int, float] = SAMPLE_RATE,
    weights: Union[int, float, str, np.ndarray] = "equal",
    envelope_fun: Union[str, Callable[[int], np.ndarray]] = "linear",
) -> np.ndarray:
    """
    Additive synthesis for a single note

    Parameters
    ----------
    freqs: Union[int, float, np.ndarray]
        Frequencies of the spectrum of the note.
    duration: float
        Duration of the note in seconds.
    samplerate: int, float
        Sample rate of the note.
    weights: int, float, str or np.ndarray
        Additive weights for the frequencies. If "equal",
        the weights will be 1 / len(freqs) for each frequency.
    envelope_fun: callable, str
        Function for defining the amplitude envelope. If a string,
        it must be "linear" or "exp". See `lin_in_lin_out`
        and `exp_in_exp_out`.

    Returns
    -------
    output: np.ndarray
        The signal corresponding to the note defined by the
        given frequencies, duration, weights and envelop.
        The length of the output is `int(np.round(duration * samplerate))`.
    """
    if isinstance(freqs, (int, float, np.integer, np.floating)):
        freqs = np.array([freqs], dtype=DTYPE)

    if not isinstance(freqs, np.ndarray):
        raise ValueError(
            "`freqs` should be a numpy array, an int or a float "
            f"but is {type(freqs)}"
        )

    if duration < 0 or samplerate < 0:
        raise ValueError(
            "`duration` and `samplerate` must be larger than 0 "
            f"but are {duration}, {samplerate}, respectively"
        )

    if isinstance(weights, (int, float, np.integer, np.floating)):
        weights = np.array([weights], dtype=DTYPE)

    elif isinstance(weights, str):
        if weights == "equal":
            weights = np.ones(len(freqs), dtype=DTYPE) / len(freqs)
        else:
            raise ValueError(
                f"If `weights` is a string, it must be 'equal' but is {weights}"
            )

    if not isinstance(weights, np.ndarray):
        raise ValueError(
            "`weights` should be a numpy array, a number (int or float) or "
            f"a string with value 'equal' but is {type(weights)}"
        )

    if len(freqs) != len(weights):
        raise ValueError("`len(freqs)` should be equal to `len(weights)`")

    freqs = freqs.reshape(-1, 1)
    weights = weights.reshape(-1, 1)

    if isinstance(envelope_fun, str):
        if envelope_fun == "linear":
            envelope_fun = lin_in_lin_out
        elif envelope_fun == "exp":
            envelope_fun = exp_in_exp_out
        else:
            raise ValueError(
                "If `envelope_fun` is a string, it must be 'linear' or 'exp', "
                f"but is {envelope_fun}"
            )

    if not callable(envelope_fun):
        raise ValueError('`envelope_fun` must be "linear", "exp" or a callable')

    num_frames = int(np.round(duration * samplerate))
    envelope = envelope_fun(num_frames)
    x = np.linspace(0, duration, num=num_frames)
    y = weights * np.sin(TWO_PI * freqs * x)

    output = y.sum(0) * envelope

    return output


class DistributedHarmonics(object):
    def __init__(
        self,
        n_harmonics: int,
        weights: Union[np.ndarray, str] = "equal",
    ) -> None:
        self.n_harmonics = n_harmonics
        self.weights = weights

        if self.weights == "equal":
            self.weights = 1.0 / (self.n_harmonics + 1) * np.ones(self.n_harmonics + 1)

        self._overtones = np.arange(1, self.n_harmonics + 2)

    def __call__(self, freq: float) -> Tuple[np.ndarray]:
        return self._overtones * freq, self.weights


class ShepardTones(object):
    """
    Generate Shepard Tones
    """

    def __init__(
        self,
        min_freq: Union[float, int] = 77.8,
        max_freq: Union[float, int] = 2349,
    ) -> None:
        self.min_freq = min_freq
        self.max_freq = max_freq

        x_freq = np.linspace(self.min_freq, self.max_freq, 1000)

        weights = np.hanning(len(x_freq) + 2) + 0.001
        weights /= max(weights)

        self.shepard_weights_fun = interp1d(
            x=x_freq,
            y=weights[1:-1],
            bounds_error=False,
            fill_value=weights.min(),
        )

    def __call__(self, freq) -> Tuple[np.ndarray]:
        min_freq = self.min_f(freq)

        freqs = 2 ** np.arange(5) * min_freq

        return freqs, self.shepard_weights_fun(freqs)

    def min_f(self, freq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        n = np.floor(np.log2(freq) - np.log2(self.min_freq))
        return freq / (2**n)

    def max_f(self, freq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        n = np.floor(np.log2(self.max_freq) - np.log2(freq))
        return freq * (2**n)


def synthesize(
    note_info: Union[ScoreLike, PerformanceLike, np.ndarray],
    samplerate: int = SAMPLE_RATE,
    envelope_fun: str = "linear",
    tuning: Union[str, Callable] = "equal_temperament",
    tuning_kwargs: Dict[str, Any] = {"a4": A4},
    harmonic_dist: Optional[Union[str, int]] = None,
    bpm: Union[float, np.ndarray, Callable] = 60,
) -> np.ndarray:
    """
    Synthesize a partitura object with note information
    using additive synthesis


    Parameters
    ----------
    note_info : ScoreLike, PerformanceLike or np.ndarray
        A partitura object with note information.
    samplerate: int
        The sample rate of the audio file in Hz.
    envelope_fun: {"linear", "exp" }
        The type of envelop to apply to the individual sine waves.
    tuning: {"equal_temperament", "natural"} or callable.
        The tuning system to use. If the value is "equal_temperament",
        12 tone equal temperament implemented in `midi_pitch_to_frequency` will
        be used. If the value is "natural", the function
        `midi_pitch_to_tempered_frequency` will be used. Note that
        `midi_pitch_to_tempered_frequency` computes the intervals (and thus,
        frequencies) with respect to a reference note (A4 by default) and uses the
        interval ratios specified by `FIVE_LIMIT_INTERVAL_RATIOS`. See
        the documentation of these functions for more information. If a callable
        is provided, function should get MIDI pitch as input and return
        frequency in Hz as output.
    tuning_kwargs : dict
        Dictionary of keyword arguments to be passed to the tuning function
        specified in  `tuning`. See `midi_pitch_to_tempered_frequency` and
       `midi_pitch_to_frequency` for more information on their keyword arguments.
    harmonic_dist : int,  "shepard" or None (optional)
        Distribution of harmonics. If an integer, it is the number
        of harmonics to be considered. If "shepard", it uses Shepard tones.
        Default is None (i.e., only consider the fundamental frequency)
    bpm : float, np.ndarray or callable
        The bpm to render the output (if the input is a score-like object).
        See `partitura.utils.music.performance_notearray_from_score_notearray`
        for more information on this parameter.

    Returns
    -------
    audio_signal : np.ndarray
       Audio signal as a 1D array.
    """

    note_array = ensure_notearray(note_info)

    onset_unit, duration_unit = get_time_units_from_note_array(note_array)
    if np.min(note_array[onset_unit]) <= 0:
        note_array[onset_unit] = note_array[onset_unit] + np.min(note_array[onset_unit])

    # If the input is a score, convert score time to seconds
    if onset_unit != "onset_sec":
        pnote_array = performance_notearray_from_score_notearray(
            snote_array=note_array,
            bpm=bpm,
        )
        onsets = pnote_array["onset_sec"]
        offsets = pnote_array["onset_sec"] + pnote_array["duration_sec"]
        duration = pnote_array["duration_sec"]
    else:
        onsets = note_array["onset_sec"]
        offsets = note_array["onset_sec"] + note_array["duration_sec"]
        duration = note_array["duration_sec"]

    pitch = note_array["pitch"]

    # Duration of the piece
    piece_duration = offsets.max()

    # Number of frames
    num_frames = int(np.round(piece_duration * samplerate))

    # Initialize array containing audio
    audio_signal = np.zeros(num_frames, dtype="float")

    # Initialize the time axis
    x = np.linspace(0, piece_duration, num=num_frames)

    # onsets in frames (i.e., indices of the `audio` array)
    onsets_in_frames = np.searchsorted(x, onsets, side="left")

    # frequency of the note in herz
    if tuning == "equal_temperament":
        freq_in_hz = midi_pitch_to_frequency(pitch, **tuning_kwargs)
    elif tuning == "natural":
        freq_in_hz = midi_pitch_to_tempered_frequency(pitch, **tuning_kwargs)
    elif callable(tuning):
        freq_in_hz = tuning(pitch, **tuning_kwargs)

    else:
        raise ValueError(
            "`tuning` must be 'equal_temperament', 'natural' or a callable"
        )

    if harmonic_dist is None:

        def harmonic_dist(x):
            return x, 1

    elif isinstance(harmonic_dist, int):
        harmonic_dist = DistributedHarmonics(harmonic_dist)

    elif isinstance(harmonic_dist, str):
        if harmonic_dist in ("shepard",):
            harmonic_dist = ShepardTones()

    for f, oif, dur in zip(freq_in_hz, onsets_in_frames, duration):
        freqs, weights = harmonic_dist(f)

        note = additive_synthesis(
            freqs=freqs,
            duration=dur,
            samplerate=samplerate,
            weights=weights,
            envelope_fun=envelope_fun,
        )
        idx = slice(oif, oif + len(note))
        audio_signal[idx] += note

    # normalization term
    # TODO: Non-linear normalization?
    norm_term = max(audio_signal.max(), abs(audio_signal.min()))

    # normalize audio
    audio_signal /= norm_term

    return audio_signal
