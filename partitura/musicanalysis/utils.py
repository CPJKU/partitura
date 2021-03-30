import numpy as np


def prepare_notearray(notearray):
    # * check whether notearray is a structured array
    # * check whether it has pitch/onset/duration fields
    # * return a copy of pitch/onset/duration fields with added id field
    if notearray.dtype.fields is None:
        raise ValueError("`notearray` must be a structured numpy array")

    fields = ("pitch", "onset", "duration")
    time_units = ("", "_div", "_quarter", "_beat", "_sec")
    onset_field = None
    duration_field = None

    for unit in time_units:
        onset_field = f"onset{unit}"
        if onset_field in notearray.dtype.names:
            break
    for unit in time_units:
        duration_field = f"duration{unit}"
        if duration_field in notearray.dtype.names:
            break
    # if field not in notearray.dtype.names:
    #     raise ValueError('Input array does not contain required field {0}'.format(field))

    dtypes = dict(notearray.dtype.descr)
    new_dtype = [
        (n, dtypes[m]) for n, m in zip(fields, ["pitch", onset_field, duration_field])
    ] + [("id", "i4")]

    return np.fromiter(
        zip(
            notearray["pitch"],
            notearray[onset_field],
            notearray[duration_field],
            np.arange(len(notearray)),
        ),
        dtype=new_dtype,
    )
