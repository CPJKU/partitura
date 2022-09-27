"""
Synthesize Partitura Part or Note array to wav using additive synthesis

TODO
* Add other tuning systems

"""
from typing import Union, Tuple

import numpy as np

from scipy.interpolate import interp1d
from scipy.io import wavfile

from partitura.utils.music import (
    midi_pitch_to_frequency,
    A4,
    get_time_units_from_note_array,
    ensure_notearray,
)


TWO_PI = 2 * np.pi
SAMPLE_RATE = 44100
DTYPE = float

NATURAL_INTERVAL_RATIOS = {
    0: 1,
    1: 16 / 15,  # 15/14, 11/10
    2: 8 / 7,  # 9/8, 10/9, 12/11, 13/14
    3: 6 / 5,  # 7/6,
    4: 5 / 4,
    5: 4 / 3,
    6: 7 / 5,  # 13/9,
    7: 3 / 2,
    8: 8 / 5,
    9: 5 / 3,
    10: 7 / 4,  # 13/7
    11: 15 / 8,
    12: 2,
}


def midi_pitch_to_natural_frequency(
    midi_pitch: Union[int, float, np.ndarray],
    a4: Union[int, float] = A4,
    natural_interval_ratios: dict = NATURAL_INTERVAL_RATIOS,
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


    TODO
    ----
    * compute intervals with given reference pitch.
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


def exp_in_exp_out(
    num_frames: int,
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
    envelope = np.ones(num_frames, dtype=DTYPE)
    # number of frames for decay
    decay_frames = np.minimum(num_frames // 10, 1000)
    # number of frames for attack
    attack_frames = np.minimum(num_frames // 100, 1000)
    # Compute envelope
    envelope[-decay_frames:] = np.exp(-np.linspace(0, 100, decay_frames)).astype(DTYPE)
    envelope[:attack_frames] = np.exp(np.linspace(-100, 0, attack_frames)).astype(DTYPE)

    return envelope


def lin_in_lin_out(num_frames: int) -> np.ndarray:
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
    envelope = np.ones(num_frames, dtype=DTYPE)
    # Number of frames for decay
    decay_frames = np.minimum(num_frames // 10, 1000)
    # number of frames for attack
    attack_frames = np.minimum(num_frames // 100, 1000)
    # Compute envelope
    envelope[-decay_frames:] = np.linspace(1, 0, decay_frames, dtype=DTYPE)
    envelope[:attack_frames] = np.linspace(0, 1, attack_frames, dtype=DTYPE)
    return envelope


def additive_synthesis(
    freqs: Union[int, float, np.ndarray],
    duration: float,
    samplerate: Union[int, float] = SAMPLE_RATE,
    weights: Union[int, float, str, np.ndarray] = "equal",
    envelope_fun="linear",
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

    """
    if isinstance(freqs, (int, float)):
        freqs = [freqs]

    if isinstance(weights, (int, float)):
        weights = [weights]

    elif weights == "equal":
        weights = np.ones(len(freqs), dtype=DTYPE) / len(freqs)

    freqs = np.array(freqs).reshape(-1, 1)
    weights = np.array(weights).reshape(-1, 1)

    if envelope_fun == "linear":
        envelope_fun = lin_in_lin_out
    elif envelope_fun == "exp":
        envelope_fun = exp_in_exp_out
    else:
        if not callable(envelope_fun):
            raise ValueError('`envelope_fun` must be "linear", "exp" or a callable')

    num_frames = int(np.round(duration * samplerate))
    envelope = envelope_fun(num_frames)
    x = np.linspace(0, duration, num=num_frames)
    output = weights * np.sin(TWO_PI * freqs * x)

    return output.sum(0) * envelope


class DistributedHarmonics(object):
    def __init__(self, n_harmonics: int, weights: Union[np.ndarray, str] = "equal"):

        self.n_harmonics = n_harmonics
        self.weights = weights

        if self.weights == "equal":
            self.weights = 1.0 / (self.n_harmonics + 1) * np.ones(self.n_harmonics + 1)

        self._overtones = np.arange(1, self.n_harmonics + 2)

    def __call__(self, freq: float) -> Tuple[np.ndarray, np.ndarray]:

        return self._overtones * freq, self.weights


class ShepardTones(object):
    """
    Generate Shepard Tones
    """

    def __init__(
        self, min_freq: Union[float, int] = 77.8, max_freq: Union[float, int] = 2349
    ):

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

    def __call__(self, freq):

        min_freq = self.min_f(freq)

        freqs = 2 ** np.arange(5) * min_freq

        return freqs, self.shepard_weights_fun(freqs)

    def min_f(self, freq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        n = np.floor(np.log2(freq) - np.log2(self.min_freq))
        return freq / (2 ** n)

    def max_f(self, freq):
        n = np.floor(np.log2(self.max_freq) - np.log2(freq))

        return freq * (2 ** n)


def check_instance(fn):
    """
    Checks if input is Partitura part object or structured array

    """
    from partitura.score import Part, PartGroup, Score
    from partitura.performance import PerformedPart, Performance

    if isinstance(fn, (Part, PartGroup, PerformedPart, Score, Performance)):
        return True
    elif isinstance(fn, list) and isinstance(fn[0], Part):
        return True
    elif isinstance(fn, np.ndarray):
        return False
    else:
        raise TypeError("The file type is not supported.")


def synthesize(
    note_info,
    out_fn=None,
    samplerate=SAMPLE_RATE,
    envelope_fun="linear",
    tuning="equal_temperament",
    harmonic_dist=None,
    bpm: Union[float, int] = 60,
) -> np.ndarray:
    """
    Synthesize_data from part or note array.


    Parameters
    ----------
    note_info : Part, PerformedPart or structured array
        A partitura Part Object (or group part or part list) or a Note array.
    out_fn : str (optional)
        filname of the output audio file
    envelope_fun: str
        The type of envelop to apply to the individual sines
    harmonic_dist : int, str or None (optional)
        Default is None.
    bpm : int
        The bpm (if the input is a score)

    Returns
    -------
    audio_signal : np.ndarray
       Audio signal as a 1D array.
    """
    if check_instance(note_info):
        note_array = ensure_notearray(note_info)
    else:
        note_array = note_info

    onset_unit, duration_unit = get_time_units_from_note_array(note_array)
    if np.min(note_array[onset_unit]) <= 0:
        note_array[onset_unit] = note_array[onset_unit] + np.min(note_array[onset_unit])

    # If the input is a score, convert score time to seconds
    if onset_unit != "onset_sec":
        beat2sec = 60 / bpm
        onsets = note_array[onset_unit] * beat2sec
        offsets = (note_array[onset_unit] + note_array[duration_unit]) * beat2sec
        duration = note_array[duration_unit] * beat2sec
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
        freq_in_hz = midi_pitch_to_frequency(pitch)
    elif tuning == "natural":
        freq_in_hz = midi_pitch_to_natural_frequency(pitch)

    if harmonic_dist is None:

        def harmonic_dist(x):
            return x, 1

    elif isinstance(harmonic_dist, int):

        harmonic_dist = DistributedHarmonics(harmonic_dist)

    elif isinstance(harmonic_dist, str):
        if harmonic_dist in ("shepard",):
            harmonic_dist = ShepardTones()

    for (f, oif, dur) in zip(freq_in_hz, onsets_in_frames, duration):

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

    if out_fn is not None:
        # Write audio signal
        wavfile.write(out_fn, samplerate, audio_signal)

    return audio_signal
