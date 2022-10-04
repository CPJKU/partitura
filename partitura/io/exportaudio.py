"""
Synthesize Partitura object to wav using additive synthesis
"""
from typing import Union, Optional
import numpy as np
from scipy.io import wavfile

from partitura.score import ScoreLike
from partitura.performance import PerformanceLike

from partitura.utils.synth import synthesize, SAMPLE_RATE

from partitura.utils.misc import PathLike

__all__ = ["save_wav"]


def save_wav(
    input_data: Union[ScoreLike, PerformanceLike, np.ndarray],
    out: Optional[PathLike] = None,
    samplerate=SAMPLE_RATE,
    envelope_fun="linear",
    tuning="equal_temperament",
    harmonic_dist: Optional[Union[str, int]] = None,
    bpm: Union[float, int] = 60,
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
    envelope_fun: {"linear", "exp" }
        The type of envelop to apply to the individual sine waves.
    tuning: {"equal_temperament", "natural"}
    harmonic_dist : int,  "shepard" or None (optional)
        Distribution of harmonics. If an integer, it is the number
        of harmonics to be considered. If "shepard", it uses Shepard tones.
        Default is None (i.e., only consider the fundamental frequency)
    bpm : int
        The bpm to render the output (if the input is a score-like object)

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
        harmonic_dist=harmonic_dist,
        bpm=bpm,
    )

    if out is not None:
        # Write audio signal
        wavfile.write(out, samplerate, audio_signal)
    else:
        return audio_signal
