"""
Synthesize Partitura Part or Note array to wav using additive synthesis

"""
import numpy as np

from scipy.interpolate import interp1d
from scipy.io import wavefile

from partitura.utils.music import (midi_pitch_to_frequency, A4, get_time_units_from_note_array, ensure_notearray,)


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


def midinote2naturalfreq(
    midi_pitch,
    a4=A4,
    natural_interval_ratios=NATURAL_INTERVAL_RATIOS,
):
    octave = (midi_pitch // 12) - 1

    aref = 69.0 - 12.0 * (4 - octave)

    aref_freq = a4 / (2.0 ** ((4 - octave)))

    interval = midi_pitch - aref

    if isinstance(interval, (int, float)):
        interval = np.array([interval], dtype=int)

    ratios = np.zeros_like(interval)
    for i, itv in enumerate(interval):
        ratios[i] = natural_interval_ratios[abs(itv)] ** (1 if itv >= 0 else -1)

    freqs = aref_freq * ratios

    if isinstance(midi_pitch, (int, float)):
        freqs = float(freqs)
    return freqs


def exp_in_exp_out(num_frames, attack_frames, decay_frames):
    """
    Sound envelope with exponential attack and decay
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


def lin_in_lin_out(num_frames):
    """
    Sound envelope with linear attack and decay
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
    freqs,
    duration,
    samplerate=SAMPLE_RATE,
    weights="equal",
    envelope_fun="linear",
):
    """
    Additive synthesis
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

    num_frames = int(np.round(duration * SAMPLE_RATE))
    envelope = envelope_fun(num_frames)
    x = np.linspace(0, duration, num=num_frames)
    output = weights * np.sin(TWO_PI * freqs * x)

    return output.sum(0) * envelope


class DistributedHarmonics(object):
    def __init__(self, n_harmonics, weights="equal"):

        self.n_harmonics = n_harmonics
        self.weights = weights

        if self.weights == "equal":
            self.weights = 1.0 / (self.n_harmonics + 1) * np.ones(self.n_harmonics + 1)

        self._overtones = np.arange(1, self.n_harmonics + 2)

    def __call__(self, freq):

        return self._overtones * freq, self.weights


class ShepardTones(object):
    """
    Generate Shepard Tones
    """

    def __init__(self, min_freq=77.8, max_freq=2349):

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

    def min_f(self, freq):
        n = np.floor(np.log2(freq) - np.log2(self.min_freq))

        return freq / (2 ** n)

    def max_f(self, freq):
        n = np.floor(np.log2(self.max_freq) - np.log2(freq))

        return freq * (2 ** n)


def check_instance(fn):
    """
    Checks if input is Partitura part object or structured array

    """
    from partitura.score import Part, PartGroup
    from partitura.performance import PerformedPart

    if isinstance(fn, (Part, PartGroup, PerformedPart)):
        return True
    elif isinstance(fn, list) and isinstance(fn[0], Part):
        return True
    elif isinstance(fn, np.ndarray):
        return False
    else:
        raise TypeError("The file type is not supported.")


def synthesize_data(
    in_fn,
    out_fn=None,
    samplerate=SAMPLE_RATE,
    envelope_fun="linear",
    tuning="equal_temperament",
    harmonic_dist=None,
    bpm=60,
):
    """
    Synthesize_data from part or note array.


    Parameters
    ----------
    in_fn : Part object or structured array
        A partitura Part Object (or group part or part list) or a Note array.
    out_fn : str (optional)
        filname of the output audio file 
    envelope_fun: str
        The type of envelop to apply to the individual sines
    harmonic_dist : int or str
        Default is None. Option is shepard.
    bpm : int
        The bpm (if the input is a score)
    """
    if check_instance(in_fn):
        note_array = ensure_notearray(in_fn)
    else:
        note_array = in_fn

    onset_unit, duration_unit = get_time_units_from_note_array(note_array)
    if np.min(note_array[onset_unit]) <= 0:
        note_array[onset_unit] = note_array[onset_unit] + np.min(note_array[onset_unit])

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

    piece_duration = offsets.max()

    # Number of frames
    num_frames = int(np.round(piece_duration * samplerate))

    # Initialize array containing audio
    audio = np.zeros(num_frames, dtype="float")

    # Initialize the time axis
    x = np.linspace(0, piece_duration, num=num_frames)

    # onsets in frames (i.e., indices of the `audio` array)
    onsets_in_frames = np.digitize(onsets, x)

    # frequency of the note in herz
    if tuning == "equal_temperament":
        freq_in_hz = midi_pitch_to_frequency(pitch)
    elif tuning == "natural":
        freq_in_hz = midinote2naturalfreq(pitch)

    if harmonic_dist is None:

        def harmonic_dist(x):
            return x, 1

    elif isinstance(harmonic_dist, int):

        harmonic_dist = DistributedHarmonics(harmonic_dist)

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

        audio[idx] += note

    # normalization term
    # TODO: Non-linear normalization?
    norm_term = max(audio.max(), abs(audio.min()))

    # normalize audio
    audio /= norm_term

    if out_fn is not None:
        amplitude = np.iinfo(float).max
        audio *= amplitude
        wavefile.write(out_fn, samplerate, audio)

    return audio
