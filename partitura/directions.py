#!/usr/bin/env python

import logging

logging.basicConfig()
LOGGER = logging.getLogger(__name__)

try:
    from .ply_annotation_parser import parse_words
    # This suppresses warnings from ply (which are plenty, but not all
    # serious). If you want to work on the ply annotation parser, you should set the
    # log level to a lower value.
    logging.getLogger('ply').setLevel(logging.ERROR)
# except ImportError:
except ModuleNotFoundError:
    LOGGER.warning('One or more packages needed for annotation parsing were not found: Using a dummy parser')

    from .scoreontology import Words

    def parse_words(words):
        return Words(words)


