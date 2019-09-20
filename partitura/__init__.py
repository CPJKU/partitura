"""

"""

import pkg_resources

from .importmusicxml import load_musicxml
from .exportmusicxml import save_musicxml
from .importmidi import load_midi
from .exportmidi import save_midi
from .directions import parse_words
from . import musicanalysis

# define a version variable
__version__ = pkg_resources.get_distribution("partitura").version

