import functools
import os
import warnings

from typing import Union, Callable, Dict, Any, Iterable

# Recommended by PEP 519
PathLike = Union[str, bytes, os.PathLike]


def get_document_name(filename: PathLike) -> str:
    """
    Get the name of a document.

    Parameters
    ----------
    filename : PathLike
        The path of the file

    Returns
    -------
    doc_name : str
        The name of the document
    """
    doc_name = str(os.path.basename(os.path.splitext(filename)[0]))
    return doc_name


def deprecated_alias(**aliases: str) -> Callable:
    """
    Decorator for aliasing deprecated function and method arguments.

    Use as follows:

    @deprecated_alias(old_arg='new_arg')
    def myfunc(new_arg):
        ...

    Notes
    -----
    Taken from https://stackoverflow.com/a/49802489 by user user2357112.
    This code is re-distributed as (Licence)
    """

    def deco(f: Callable):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco


def deprecated_parameter(*deprecated_kwargs: str) -> Callable:
    """
    Decorator for deprecating function and method arguments.

    Use as follows:

    @deprecated_parameter("old_argument1", "old_argument2")
    def func(new_arg):
        ...
    """

    def deco(f: Callable):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            to_be_deprecated(f.__name__, kwargs, deprecated_kwargs)
            return f(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(
    func_name: str,
    kwargs: Dict[str, Any],
    aliases: Dict[str, str],
) -> None:
    """
    Helper function for renaming deprecated function arguments.
    This function edits the dictionary of keyword arguments in-place.

    Parameters
    ----------
    func_name : str
        Name of the function which keyword arguments have been deprecated.
    kwargs : dictionary
        Dictionary of keyword arguments to be passed to the function
    aliases: dictionary
        Dictionary specifying the aliases of the deprecated keyword arguments.


    Notes
    -----
    Taken from https://stackoverflow.com/a/49802489 by user user2357112.
    """
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError(
                    f"{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is deprecated, use {new} instead."
                )
            warnings.warn(
                message=(
                    f"`{alias}` is deprecated as an argument to `{func_name}`; use"
                    f" `{new}` instead."
                ),
                category=DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(alias)


def to_be_deprecated(
    func_name: str,
    kwargs: Dict[str, Any],
    deprecated_kwargs: Iterable[str],
) -> None:
    """
    Helper function for deprecating function arguments.
    This function edits the dictionary of keyword arguments in-place.

    Parameters
    ----------
    func_name : str
        Name of the function which keyword arguments have been deprecated.
    kwargs : dictionary
        Dictionary of keyword arguments to be passed to the function
    deprecated_kwargs: Iterable[str]
        An iterable specifiying the parameters to be deprecated.
    """

    for deprecated_kwarg in deprecated_kwargs:
        if deprecated_kwarg in kwargs:
            # raise warning
            warnings.warn(
                message=(
                    f"`{deprecated_kwarg}` is a deprecatd argument of `{func_name}`"
                    " and will be ignored."
                ),
                category=DeprecationWarning,
                stacklevel=3,
            )
            # Remove deprecated kwarg from kwargs
            kwargs.pop(deprecated_kwarg)
