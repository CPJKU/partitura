"""
Utilities for parsing files

TODO:
----

* Add support for MIDI files
"""

# File types
from . import matchfile
from . import musicxml
from . import wormfile
# from . import midi

# Score ontology
from . import scoreontology

# Utilities
from . import annotation_parser
from . import instrument_assignment
from . import sparse_datafiles
from . import sparse_feature_extraction

# MIDI backend
# from . import midi_backend
