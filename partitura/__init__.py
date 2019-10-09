"""

"""

import pkg_resources

from .importmusicxml import load_musicxml
from .exportmusicxml import save_musicxml
from .importmidi import load_midi
from .exportmidi import save_midi
from .directions import parse_direction
from .display import show
from . import musicanalysis

# define a version variable
__version__ = pkg_resources.get_distribution("partitura").version

#: An example MusicXML file for didactic purposes  
EXAMPLE_MUSICXML = pkg_resources.resource_filename("partitura", 'assets/score_example.musicxml')

__all__ = ['load_musicxml', 'save_musicxml', 'load_midi', 'show', 'EXAMPLE_MUSICXML']
