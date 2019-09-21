"""

"""

import pkg_resources

from .importmusicxml import load_musicxml
from .exportmusicxml import save_musicxml
from .importmidi import load_midi
from .directions import parse_words
from . import musicanalysis

# define a version variable
__version__ = pkg_resources.get_distribution("partitura").version

__all__ = ['load_musicxml', 'save_musicxml', 'load_midi', 'parse_words']
