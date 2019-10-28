#!/usr/bin/env python

"""This module defines a function "show" that creates a rendering of one
or more parts or partgroups and opens it using the desktop default
application.

"""

import platform
import logging
import os
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryFile

from partitura import save_musicxml
from partitura.io.musescore import show_musescore

LOGGER = logging.getLogger(__name__)

__all__ = ['show']

# def ly_install_msg():
#     """Issue a platform specific installation suggestion for lilypond
    
#     """
#     if platform.system() == 'Linux':
#         s = ('Is lilypond installed? On debian based '
#              'installations you can install it using the '
#              'command "sudo apt install lilypond"')
#     elif platform.system() == 'Darwin':
#         s = ('Is lilypond installed? Lilypond can be '
#              'installed using brew ( https://brew.sh/ ) '
#              'using the command "brew cask install lilypond"')
#     elif platform.system() == 'Windows':
#         s = ('Is lilypond installed? It can be downloaded from '
#              'http://lilypond.org/')
#     return s

def show(part, out_fmt='png'):
    """Show a rendering of one or more parts or partgroups using the
    desktop default application.

    Rendering is first attempted through musecore, and if that fails
    through lilypond. If that also fails the function returns without
    raising an exception.

    Parameters
    ----------
    part : :class:`partitura.score.Part` or :class:`partitura.score.PartGroup` or list of these
        The score content to be displayed
    out_fmt : {'png', 'pdf'}, optional
        The image format of the rendered material
    
    """
    
    img_fn = show_musescore(part, out_fmt)
    
    if img_fn is None or not os.path.exists(img_fn):
        img_fn =  show_lilypond(part, out_fmt)
        if img_fn is None or not os.path.exists(img_fn):
            return None
    
    # NOTE: the temporary image file will not be deleted.
    if platform.system() == 'Linux':
        subprocess.call(['xdg-open', img_fn])
    elif platform.system() == 'Darwin':
        subprocess.call(['open', img_fn])
    elif platform.system() == 'Windows':
        os.startfile(img_fn)


def show_lilypond(part, out_fmt='png'):
    if out_fmt not in ('png', 'pdf'):
        print('warning: unsupported output format')
        return

    prvw_sfx = '.preview.{}'.format(out_fmt)
    
    with TemporaryFile() as xml_fh, \
         NamedTemporaryFile(suffix=prvw_sfx, delete=False) as img_fh:

        # save part to musicxml in file handle xml_fh
        save_musicxml(part, xml_fh)
        # rewind read pointer of file handle before we pass it to musicxml2ly
        xml_fh.seek(0)

        img_stem = img_fh.name[:-len(prvw_sfx)]

        # convert musicxml to lilypond format (use stdout pipe)
        cmd1 = ['musicxml2ly', '-o-', '-']
        try:
            ps1 = subprocess.run(cmd1, stdin=xml_fh, stdout=subprocess.PIPE)
            if ps1.returncode != 0:
                LOGGER.error('Command {} failed with code {}'
                             .format(cmd1, ps1.returncode))
                return
        except FileNotFoundError as f:
            LOGGER.error('Executing "{}" returned  {}.'
                         .format(' '.join(cmd1), f))
            return
        
        # convert lilypond format (read from pipe of ps1) to image, and save to
        # temporary filename
        cmd2 = ['lilypond', '--{}'.format(out_fmt),
                '-dno-print-pages', '-dpreview',
                '-o{}'.format(img_stem), '-']
        try:
            ps2 = subprocess.run(cmd2, input=ps1.stdout)
            if ps2.returncode != 0:
                LOGGER.error('Command {} failed with code {}'
                             .format(cmd2, ps2.returncode))
                return
        except FileNotFoundError as f:
            LOGGER.error('Executing "{}" returned {}.'
                         .format(' '.join(cmd2), f))
            return

        return img_fh.name
