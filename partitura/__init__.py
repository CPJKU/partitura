"""

"""

import pkg_resources

from .importmusicxml import load_musicxml
from .exportmusicxml import save_musicxml
from .importmidi import load_score_midi, load_performance_midi
from .exportmidi import save_score_midi, save_performance_midi
from .importmatch import load_match
from .display import show
from . import musicanalysis

# define a version variable
__version__ = pkg_resources.get_distribution("partitura").version

#: An example MusicXML file for didactic purposes  
EXAMPLE_MUSICXML = pkg_resources.resource_filename("partitura", 'assets/score_example.musicxml')

__all__ = ['load_musicxml', 'save_musicxml',
           'load_score_midi', 'save_score_midi',
           'load_performance_midi', 'save_performance_midi',
           'show', 'EXAMPLE_MUSICXML']
