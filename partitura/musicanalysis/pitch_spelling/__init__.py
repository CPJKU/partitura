from .ps13 import ps13s1


def estimate_spelling(note_array, method='ps13s1', *args, **kwargs):

    if method == 'ps13s1':
        ps = ps13s1

    step, alter, octave = ps(note_array, *args, **kwargs)

    return step, alter, octave
