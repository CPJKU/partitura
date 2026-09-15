#!/usr/bin/env python
"""
Builds the optional `partitura._symusic_midi` fast MIDI-import extension.

All other package metadata lives in pyproject.toml; this file exists only
to declare the optional C++ extension, since setuptools cannot express
`ext_modules` declaratively. If the vendored minimidi submodule is missing
or the extension fails to compile, the build falls back to the pure-Python
`mido` backend at import time -- see partitura/io/importmidi.py.
"""
import os

from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))


def _fast_midi_extension():
    minimidi_include = os.path.join(HERE, "3rdparty", "minimidi", "include")
    if not os.path.isdir(minimidi_include):
        # Submodule not checked out (e.g. `git submodule update --init`
        # was not run) -- skip the optional extension entirely.
        return [], {}

    try:
        from pybind11.setup_helpers import Pybind11Extension, build_ext
    except ImportError:
        return [], {}

    ext = Pybind11Extension(
        "partitura._symusic_midi",
        sources=[os.path.join("native", "fastmidi.cpp")],
        include_dirs=[minimidi_include],
        cxx_std=20,
        optional=True,
    )
    return [ext], {"build_ext": build_ext}


ext_modules, cmdclass = _fast_midi_extension()

setup(ext_modules=ext_modules, cmdclass=cmdclass)
