#!/usr/bin/env python

import platform
import logging
import os
import subprocess
from tempfile import NamedTemporaryFile, TemporaryFile

from partitura.exportmusicxml import save_musicxml
LOGGER = logging.getLogger(__name__)

#musicxml2ly -o- out.xml | lilypond --png -dno-print-pages -dpreview -o/tmp/jo - ;xdg-open /tmp/jo.preview.png

def show(part, out_fmt='png'):
    """Show a rendering of one or more parts or partgroups using the
    desktop default application.

    Parameters
    ----------
    part: :class:`partitura.score.Part` or :class:`partitura.score.PartGroup` or list of these
        The score content to be displayed
    out_fmt: {'png', 'pdf'}, optional
        The image format of the rendered material
    """
    if out_fmt not in ('png', 'pdf'):
        print('warning: unsupported output format')
        return

    prvw_sfx = '.preview.{}'.format(out_fmt)

    with TemporaryFile() as xml_fh, \
         NamedTemporaryFile(suffix=prvw_sfx, delete=False) as img_fh:

        # save_musicxml(part, xml_fh.name)
        # ps1 = subprocess.run(['musicxml2ly', '-o-', xml_fh.name],
        #                      # universal_newlines=True,
        #                      stdout=subprocess.PIPE)

        save_musicxml(part, xml_fh)
        xml_fh.seek(0)
        
        ps1 = subprocess.run(['musicxml2ly', '-o-', '-'],
                             stdin=xml_fh,
                             stdout=subprocess.PIPE)

        if ps1.returncode != 0:
            print('error: musicxml2ly did not exit succesfully')
            return

        img_stem = img_fh.name[:-len(prvw_sfx)]
        ps2 = subprocess.run(['lilypond', '--{}'.format(out_fmt),
                              '-dno-print-pages', '-dpreview',
                              '-o{}'.format(img_stem), '-'],
                             input=ps1.stdout)

        if ps2.returncode != 0:
            print('error: lilypond did not exit succesfully')
            return

        if platform.system() == 'Linux':
            subprocess.call(['xdg-open', img_fh.name])
        elif platform.system() == 'Darwin':
            subprocess.call(['open', img_fh.name])
        elif platform.system() == 'Windows':
            os.startfile(img_fh.name)
