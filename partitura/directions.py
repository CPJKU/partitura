#!/usr/bin/env python

import logging

LOGGER = logging.getLogger(__name__)

try:
    from partitura.directionparser import parse_words
except ImportError:
    
    LOGGER.warning('One or more packages needed for annotation parsing were not found: Using a dummy parser')

    from partitura.score import Words

    def parse_words(words):
        return Words(words)


