"""The top level of the package contains functions to load and save
data, display rendered scores, and functions to estimate pitch
spelling, voice assignment, and key signature.

"""

import pkg_resources

from .io import load_score, load_performance, load_score_as_part, lp
from .io.musescore import load_via_musescore
from .io.importmusicxml import load_musicxml, musicxml_to_notearray
from .io.exportmusicxml import save_musicxml
from .io.importmei import load_mei
from .io.importkern import load_kern
from .io.importmidi import load_score_midi, load_performance_midi, midi_to_notearray
from .io.exportmidi import save_score_midi, save_performance_midi
from .io.importmatch import load_match
from .io.exportmatch import save_match
from .io.importnakamura import load_nakamuramatch, load_nakamuracorresp
from .io.importparangonada import load_parangonada_csv
from .io.exportparangonada import save_parangonada_csv, save_csv_for_parangonada
from .io.exportaudio import save_wav
from .display import render
from . import musicanalysis
from .musicanalysis import make_note_features, compute_note_array, full_note_array

# define a version variable
__version__ = pkg_resources.get_distribution("partitura").version

#: An example MusicXML file for didactic purposes
EXAMPLE_MUSICXML = pkg_resources.resource_filename(
    "partitura", "assets/score_example.musicxml")

EXAMPLE_MIDI = pkg_resources.resource_filename("partitura", "assets/score_example.mid")
EXAMPLE_MEI = pkg_resources.resource_filename("partitura", "assets/score_example.mei")
EXAMPLE_KERN = pkg_resources.resource_filename("partitura", "assets/score_example.krn")

__all__ = [
    "load_score",
    "load_score_as_part",
    "load_performance",
    "load_musicxml",
    "save_musicxml",
    "load_mei",
    "load_kern",
    "musicxml_to_notearray",
    "load_score_midi",
    "save_score_midi",
    "load_via_musescore",
    "load_performance_midi",
    "save_performance_midi",
    "load_match",
    "save_match",
    "load_nakamuramatch",
    "load_nakamuracorresp",
    "load_parangonada_csv",
    "save_parangonada_csv",
    "render"
]
