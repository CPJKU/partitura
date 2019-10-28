#!/usr/bin/env python

"""This module contains functionality to use the MuseScore program as a
backend for loading and rendering scores.

"""

import platform
import logging
import os
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryFile

LOGGER = logging.getLogger(__name__)

from partitura.io.importmusicxml import load_musicxml
from partitura.io.exportmusicxml import save_musicxml

class MuseScoreNotFoundException(Exception): pass
class FileImportException(Exception): pass
    
def find_musescore3():
    # # possible way to detect MuseScore... executable
    # for p in os.environ['PATH'].split(':'): 
    #     c = glob.glob(os.path.join(p, 'MuseScore*')) 
    #     if c: 
    #         print(c) 
    #         break 
            
    result = shutil.which('musescore')

    if result is None:
        result = shutil.which('mscore')

    if platform.system() == 'Linux':
        pass

    elif platform.system() == 'Darwin':

        result = shutil.which('/Applications/MuseScore 3.app/Contents/MacOS/mscore')

    elif platform.system() == 'Windows':
        pass

    return result


def load_via_musescore(fn):
    """Load a score through through the MuseScore program.

    This function attempts to load the file in MuseScore, export it as
    MusicXML, and then load the MusicXML. This should enable loading
    of all file formats that for which MuseScore has import-support
    (e.g. MIDI, and ABC, but currently not MEI).

    Parameters
    ----------
    fn : str
        Filename of the score to load

    Returns
    -------
    :class:`partitura.score.Part`, :class:`partitura.score.PartGroup`, or a list of these
        One or more part or partgroup objects

    """
    
    mscore_exec = find_musescore3()

    if not mscore_exec:

        raise MuseScoreNotFoundException()
    
    with NamedTemporaryFile(suffix='.musicxml') as xml_fh:

        cmd = [mscore_exec, '-o', xml_fh.name, fn]

        try:

            ps = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            if ps.returncode != 0:

                raise FileImportException('Command {} failed with code {}. MuseScore error messages:\n {}'
                                          .format(cmd, ps.returncode, ps.stderr.decode('UTF-8')))
        except FileNotFoundError as f:

            raise FileImportException('Executing "{}" returned  {}.'
                                      .format(' '.join(cmd), f))

        return load_musicxml(xml_fh.name)


def show_musescore(part, out_fmt, dpi=90):
    """Render a part using musescore.

    Parameters
    ----------
    part : Part
        Part to be rendered
    out_fmt : {'png', 'pdf'}
        Output image format
    dpi : int, optional
        Image resolution. This option is ignored when `out_fmt` is
        'pdf'. Defaults to 90.

    """
    mscore_exec = find_musescore3()

    if not mscore_exec:

        return None

    if out_fmt not in ('png', 'pdf'):

        LOGGER.warning('warning: unsupported output format')
        return None
    
    with NamedTemporaryFile(suffix='.musicxml') as xml_fh, \
        NamedTemporaryFile(suffix='.{}'.format(out_fmt), delete=False) as img_fh:

        save_musicxml(part, xml_fh)
        cmd = [mscore_exec, '-T', '10', '-r', '{}'.format(dpi), '-o', img_fh.name, xml_fh.name]

        try:

            ps = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ps.returncode != 0:
                LOGGER.error('Command {} failed with code {}'
                             .format(cmd, ps.returncode))
                return None

        except FileNotFoundError as f:

            LOGGER.error('Executing "{}" returned  {}. {}'
                         .format(' '.join(cmd), f))
            return None

        name, ext = os.path.splitext(img_fh.name)
        return '{}-1{}'.format(name, ext)


