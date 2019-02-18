#!/usr/bin/env python

"""
Common OS-related functionality.

"""

import subprocess
import sys
import tty
import termios
import os
import re
from collections import defaultdict
import signal
import logging
from functools import wraps

import pickle
import bz2
import gzip

from pickle import UnpicklingError

LOGGER = logging.getLogger(__name__)


def load_pyc_bz(fn):
    return pickle.load(bz2.BZ2File(fn, 'r'))


def save_pyc_bz(d, fn):
    pickle.dump(d, bz2.BZ2File(fn, 'w'), pickle.HIGHEST_PROTOCOL)


def load_pyc_gz(fn):
    return pickle.load(gzip.GzipFile(fn, 'r'))


def save_pyc_gz(d, fn):
    pickle.dump(d, gzip.GzipFile(fn, 'w'), pickle.HIGHEST_PROTOCOL)


def get_from_cache_or_compute(cache_fn, func, args=(), kwargs={}, refresh_cache=False):
    """
    If `cache_fn` exists, return the unpickled contents of that file
    (the cache file is treated as a bzipped pickle file). If this
    fails, compute `func`(*`args`), pickle the result to `cache_fn`,
    and return the result.

    Parameters
    ----------

    func : function
        function to compute

    args : tuple
        argument for which to evaluate `func`

    cache_fn : str
        file name to load the computed value `func`(*`args`) from

    refresh_cache : boolean
        if True, ignore the cache file, compute function, and store the result in the cache file

    Returns
    -------

    object

        the result of `func`(*`args`)

    """

    result = None
    if cache_fn is not None and os.path.exists(cache_fn):
        if refresh_cache:
            os.remove(cache_fn)
        else:
            try:
                result = load_pyc_bz(cache_fn)
            except UnpicklingError as e:
                LOGGER.error(('The file {0} exists, but cannot be unpickled. Is it readable? Is this a pickle file?'
                              '').format(cache_fn))
                raise e

    if result is None:
        result = func(*args, **kwargs)
        if cache_fn is not None:
            save_pyc_bz(result, cache_fn)
    return result


def debug_mode(logger):
    """
    Return True when logger is in DEBUG mode, False otherwise.

    :param logger: a Logger instance from the logging module

    :return: True or False

    """

    return logger.getEffectiveLevel() == logging.DEBUG


def init_worker():
    """
    Setup a worker to ignore signals
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# class KeyboardInterruptError(Exception): pass
# this approach doesn't really work
# since decorated functions can't be pickled


def catch_KeyboardInterrupt(fun):
    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except KeyboardInterrupt:
            raise KeyboardInterruptError()
    return wrapper


class Result(object):

    """
    A drop-in replacement for the Result object returned by
    asynchronous multiprocessing calls

    """

    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


class FakePool(object):

    """
    A drop-in replacement for multiprocessing.Pool that
    carries out jobs sequentially (useful for debugging).

    """

    def __init__(self, *args):
        pass

    def map(self, f, a):
        return list(map(f, a))

    def imap_unordered(self, f, a):
        for x in a:
            yield f(x)

    def apply_async(self, f, args=(), kwargs={}, callback=None):
        if callback:
            callback(f(*args, **kwargs))
        else:
            return Result(f(*args, **kwargs))

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


class PoolWrapper(object):

    """
    Class that can be
    Parameters
    ----------

    x : type
        Description of parameter `x`.

    Returns
    -------

    int
        Description of return value

    """

    def __init__(self, target):
        self.target = target
        try:
            functools.update_wrapper(self, target)
        except:
            pass

    def __call__(self, args):
        try:
            return self.target(*args)
        except KeyboardInterrupt:
            print('child interrupted')
            return None


def pair_files(dir_dict, remove_incomplete=True):
    """
    Pair files in directories;
    dir_dict is of form (label: directory)
    """
    result = defaultdict(dict)
    for label, directory in list(dir_dict.items()):
        for f in os.listdir(directory):
            name = os.path.splitext(f)[0]
            result[name][label] = f

    if remove_incomplete:
        labels = list(dir_dict.keys())
        for k in list(result.keys()):
            if not all([y in result[k] for y in labels]):
                del result[k]

    return result


def pair_files_new(dir_dict, remove_incomplete=True,
                   split=False, remove_parts=set()):
    """
    Pair files in directories;
    dir_dict is of form (label: directory)
    """
    result = defaultdict(dict)
    if split:
        pat = re.compile(split)
    for label, directory in list(dir_dict.items()):
        for f in os.listdir(directory):
            name = os.path.splitext(f)[0]
            if split:
                key = tuple([x for i, x in enumerate(pat.split(name))
                             if not i in remove_parts])
            else:
                key = name
            result[key][label] = f

    if remove_incomplete:
        labels = list(dir_dict.keys())
        for k in list(result.keys()):
            if not all([y in result[k] for y in labels]):
                del result[k]

    return result


def get_output_from_command(cmd):
    """Simple wrapper around popen2, to get output from a shell command"""
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, close_fds=True)
    p.stdin.close()
    result = p.stdout.readlines()
    p.stdout.close()
    return result


class _Getch:

    """
    Get a single character from standard input;
    Do not echo to the screen
    """

    def __init__(self, enc=False):
        try:
            self.impl = _GetchUnix(enc)
        except ImportError:
            self.impl = _GetchWindows(enc)

    def __call__(self):
        return self.impl()


class _GetchUnix:

    def __init__(self, enc=False):
        self.enc = enc
        if self.enc:
            import codecs
            import locale
            try:
                # Wrap stdin with an encoding-aware reader.
                _, encoding = locale.getdefaultlocale()
            except ValueError:
                encoding = 'UTF-8'
            self.stdin = codecs.getreader(encoding)(sys.stdin)
        else:
            self.stdin = sys.stdin

    def __call__(self):
        # import sys, tty, termios

        fd = self.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = self.stdin.read(1)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:

    def __init__(self, enc=False):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

get_character = _Getch()
get_character_enc = _Getch(enc=True)


def interrupt_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except KeyboardInterrupt:
            print(('\nfunction {0}.{1} was configured to intercept '
                  'KeyboardInterrupts. Press "a" to abort, or any other '
                  'key to continue.'
                  .format(f.__module__, f.__name__)))
            x = get_character()
            if x == 'a':
                raise KeyboardInterrupt
            else:
                return None
    return wrapper

if __name__ == '__main__':
    pass
