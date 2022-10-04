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

    return audio_signal
