from .krumhansl_shepard import estimate_key as ks_kid


def estimate_key(note_array, method='krumhansl', *args, **kwargs):
    """
    Estimate key of a piece

    Parameters
    ----------
    note_array : structured array
        Array containing the score
    method : ('krumhansl', 'temperley')
        Method for estimating the key. Default is 'krumhansl'.
        More options will be added in the future.
    *args, *kwargs
        Positional and Keyword arguments for the key estimation method

    Returns
    -------
    root : str
        Root of the key (key name)
    mode : str
        Mode of the key ('major' or 'minor')
    fifths: int
        Position in the circle of fifths
    """
    if method not in ('krumhansl', ):
        raise ValueError('For now the only valid method is "krumhansl"')

    if method == 'krumhansl':
        kid = ks_kid
    if method == 'temperley':
        kid = ks_kid
        if 'key_profiles' not in kwargs:
            kwargs['key_profiles'] = 'temperley'

    root, mode, fifths = kid(note_array, *args, **kwargs)

    return root, mode, fifths
