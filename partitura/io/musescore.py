#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains functionality to use the MuseScore program as a
backend for loading and rendering scores.
"""

import platform
import warnings
import glob
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, gettempdir
from typing import Optional, Union

from partitura.io.importmusicxml import load_musicxml
from partitura.io.exportmusicxml import save_musicxml
from partitura.score import Score, ScoreLike

from partitura.utils.misc import (
    deprecated_alias,
    deprecated_parameter,
    PathLike,
    concatenate_images,
    PIL_EXISTS,
)


class MuseScoreNotFoundException(Exception):
    pass


class FileImportException(Exception):
    pass


def find_musescore_version(version=4):
    """Find the path to the MuseScore executable for a specific version.
    If version is a empty string it tries to find an unspecified version of
    MuseScore which is used in some systems.
    """
    result = shutil.which(f"musescore{version}")
    if result is None:
        result = shutil.which(f"mscore{version}")
    if result is None:
        if platform.system() == "Linux":
            pass
        elif platform.system() == "Darwin":
            result = shutil.which(
                f"/Applications/MuseScore {version}.app/Contents/MacOS/mscore"
            )
        elif platform.system() == "Windows":
            result = shutil.which(
                rf"C:\Program Files\MuseScore {version}\bin\MuseScore{version}.exe"
            )

    return result


def find_musescore():
    """Find the path to the MuseScore executable.

    This function first tries to find the executable for MuseScore 4,
    then for MuseScore 3, and finally for any version of MuseScore.

    Returns
    -------
    str
        Path to the MuseScore executable

    Raises
    ------
    MuseScoreNotFoundException
        When no MuseScore executable was found
    """

    mscore_exec = find_musescore_version(version=4)
    if not mscore_exec:
        mscore_exec = find_musescore_version(version=3)
        if mscore_exec:
            warnings.warn(
                "Only Musescore 3 is installed. Consider upgrading to musescore 4."
            )
        else:
            mscore_exec = find_musescore_version(version="")
            if mscore_exec:
                warnings.warn(
                    "A unspecified version of MuseScore was found. Consider upgrading to musescore 4."
                )
            else:
                raise MuseScoreNotFoundException()
    # check if a screen is available (only on Linux)
    if "DISPLAY" not in os.environ and platform.system() == "Linux":
        raise MuseScoreNotFoundException(
            "Musescore Executable was found, but a screen is missing. Musescore needs a screen to load scores"
        )

    return mscore_exec


@deprecated_alias(fn="filename")
@deprecated_parameter("ensure_list")
def load_via_musescore(
    filename: PathLike,
    validate: bool = False,
    force_note_ids: Optional[Union[bool, str]] = True,
) -> Score:
    """Load a score through through the MuseScore program.

    This function attempts to load the file in MuseScore, export it as
    MusicXML, and then load the MusicXML. This should enable loading
    of all file formats that for which MuseScore has import-support
    (e.g. MIDI, and ABC, but currently not MEI).

    Parameters
    ----------
    filename : str
        Filename of the score to load
    validate : bool, optional
        When True the validity of the MusicXML generated by MuseScore is checked
        against the MusicXML 3.1 specification before loading the file. An
        exception will be raised when the MusicXML is invalid.
        Defaults to False.
    force_note_ids : bool, optional.
        When True each Note in the returned Part(s) will have a newly
        assigned unique id attribute. Existing note id attributes in
        the MusicXML will be discarded.

    Returns
    -------
    :class:`partitura.score.Part`, :class:`partitura.score.PartGroup`, \
or a list of these
        One or more part or partgroup objects

    """
    if filename.endswith(".mscz"):
        pass
    else:
        # open the file as text and check if the first symbol is "<" to avoid
        # further processing in case of non-XML files
        with open(filename, "r") as f:
            if f.read(1) != "<":
                raise FileImportException(
                    "File {} is not a valid XML file.".format(filename)
                )

    mscore_exec = find_musescore()

    xml_fh = os.path.splitext(os.path.basename(filename))[0] + ".musicxml"

    cmd = [mscore_exec, "-o", xml_fh, filename, "-f"]

    try:
        ps = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        if ps.returncode != 0:
            raise FileImportException(
                (
                    "Command {} failed with code {}. MuseScore " "error messages:\n {}"
                ).format(cmd, ps.returncode, ps.stderr.decode("UTF-8"))
            )
    except FileNotFoundError as f:
        raise FileImportException(
            'Executing "{}" returned  {}.'.format(" ".join(cmd), f)
        )

    score = load_musicxml(
        filename=xml_fh,
        validate=validate,
        force_note_ids=force_note_ids,
    )

    os.remove(xml_fh)

    return score


@deprecated_alias(out_fn="out", part="score_data")
def render_musescore(
    score_data: ScoreLike,
    fmt: str,
    out: Optional[PathLike] = None,
    dpi: Optional[int] = 90,
) -> Optional[PathLike]:
    """
    Render a score-like object using musescore.

    Parameters
    ----------
    score_data : ScoreLike
        Score-like object to be rendered
    fmt : {'png', 'pdf'}
        Output image format
    out : str or None, optional
        The path of the image output file, if not specified, the
        rendering will be saved to a temporary filename. Defaults to
        None.
    dpi : int, optional
        Image resolution. This option is ignored when `fmt` is
        'pdf'. Defaults to 90.

    Returns
    -------
    out : Optional[PathLike]
       Path to the output generated image (or None if no image was generated)
    """
    mscore_exec = find_musescore()

    if fmt not in ("png", "pdf"):
        warnings.warn("warning: unsupported output format")
        return None

    # with NamedTemporaryFile(suffix='.musicxml') as xml_fh, \
    #      NamedTemporaryFile(suffix='.{}'.format(fmt)) as img_fh:
    with TemporaryDirectory() as tmpdir:
        xml_fh = Path(tmpdir) / "score.musicxml"
        img_fh = Path(tmpdir) / f"score.{fmt}"

        save_musicxml(score_data, xml_fh)

        cmd = [
            mscore_exec,
            "-T",
            "10",
            "-r",
            "{}".format(int(dpi)),
            "-o",
            os.fspath(img_fh),
            os.fspath(xml_fh),
            "-f",
        ]
        try:
            ps = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if ps.returncode != 0:
                warnings.warn(
                    "Command {} failed with code {}; stdout: {}; stderr: {}".format(
                        cmd,
                        ps.returncode,
                        ps.stdout.decode("UTF-8"),
                        ps.stderr.decode("UTF-8"),
                    ),
                    SyntaxWarning,
                    stacklevel=2,
                )
                return None

        except FileNotFoundError as f:
            warnings.warn(
                'Executing "{}" returned  {}.'.format(" ".join(cmd), f),
                ImportWarning,
                stacklevel=2,
            )
            return None

        # LOGGER.error('Command "{}" returned with code {}; stdout: {}; stderr: {}'
        #              .format(' '.join(cmd), ps.returncode, ps.stdout.decode('UTF-8'),
        #                      ps.stderr.decode('UTF-8')))

        if fmt == "png":
            if PIL_EXISTS:
                # get all generated image files
                img_files = glob.glob(
                    os.path.join(img_fh.parent, img_fh.stem + "-*.png")
                )
                concatenate_images(
                    filenames=img_files,
                    out=img_fh,
                    concat_mode="vertical",
                )
            else:
                # The first image seems to be blank (MuseScore adds an empy page)
                img_fh = (img_fh.parent / (img_fh.stem + "-2")).with_suffix(
                    img_fh.suffix
                )

        if img_fh.is_file():
            if out is None:
                out = os.path.join(gettempdir(), "partitura_render_tmp.png")
            shutil.copy(img_fh, out)
            return out

        return None
