from partitura.score import Part, PartGroup
from partitura.utils import (estimate_symbolic_duration, estimate_clef_properties, key_name_to_fifths_mode, fifths_mode_to_key_name)
import warnings
import numpy as np
from typing import Union
import numpy.lib.recfunctions as rfn
from fractions import Fraction
import partitura.musicanalysis as analysis
import partitura.score as score


def create_divs_from_beats(note_array):
    duration_fractions = [Fraction(ix).limit_denominator(256) for ix in note_array["duration_beat"]]
    onset_fractions = [Fraction(ix).limit_denominator(256) for ix in note_array["onset_beat"]]
    divs = np.lcm.reduce(
        [Fraction(ix).limit_denominator(256).denominator for ix in np.unique(note_array["duration_beat"])])
    onset_divs = list(map(lambda r: int(divs * r.numerator / r.denominator), onset_fractions))
    duration_divs = list(map(lambda r: int(divs * r.numerator / r.denominator), duration_fractions))
    na_divs = np.array(list(zip(onset_divs, duration_divs)), dtype=[("onset_div", int), ("duration_div", int)])
    return rfn.merge_arrays((note_array, na_divs), flatten=True, usemask=False)


def create_part(
        ticks,
        note_array,
        key_sigs,
        part_id=None,
        part_name=None,
):
    warnings.warn("create_part", stacklevel=2)

    part = score.Part(part_id, part_name=part_name)
    part.set_quarter_duration(0, ticks)

    clef = score.Clef(
        number=1, **estimate_clef_properties(note_array["pitch"])
    )
    part.add(clef, 0)
    for t_start, name, t_end in key_sigs:
        fifths, mode = key_name_to_fifths_mode(name)
        t_start, t_end = int(t_start), int(t_end)
        part.add(score.KeySignature(fifths, mode), t_start, t_end)

    warnings.warn("add notes", stacklevel=2)

    for n in note_array:
        if n["duration_div"] > 0:
            note = score.Note(
                step=n["step"],
                octave=n["octave"],
                alter=n["alter"],
                voice=int(n["voice"] or 0),
                id=n["id"],
                symbolic_duration=estimate_symbolic_duration(n["duration_div"], ticks)
            )
        else:
            note = score.GraceNote(
                grace_type="appoggiatura",
                step=n["step"],
                octave=n["octave"],
                alter=n["alter"],
                voice=int(n["voice"] or 0),
                id=n["id"],
                symbolic_duration=dict(type="quarter"),
            )

        part.add(note, n["onset_div"], n["onset_div"] + n["duration_div"])

    if np.all(note_array["ts_beats"] == None):
        warnings.warn("No time signatures found, assuming 4/4")
        time_sigs = [[0, 4, 4]]
    else:
        time_sigs = [[0, note_array[0]["ts_beats"], note_array[0]["ts_beat_type"]]]
        for n in note_array:
            if n["ts_beats"] != time_sigs[-1][1] or n["ts_beat_type"] != time_sigs[-1][2]:
                time_sigs.append([n["onset_div"], n["ts_beats"], n["ts_beat_type"]])
    time_sigs = np.array(time_sigs)

    # for convenience we add the end times for each time signature
    ts_end_times = np.r_[time_sigs[1:, 0], np.max(note_array["onset_div"]+note_array["duration_div"])]
    time_sigs = np.column_stack((time_sigs, ts_end_times))

    warnings.warn("add time sigs and measures", stacklevel=2)

    for ts_start, num, den, ts_end in time_sigs:
        time_sig = score.TimeSignature(num.item(), den.item())
        part.add(time_sig, ts_start, ts_end)

    score.add_measures(part)

    warnings.warn("tie notes", stacklevel=2)
    # tie notes where necessary (across measure boundaries, and within measures
    # notes with compound duration)
    score.tie_notes(part)

    warnings.warn("find tuplets", stacklevel=2)
    # apply simplistic tuplet finding heuristic
    score.find_tuplets(part)

    warnings.warn("done create_part", stacklevel=2)
    return part


def note_array_to_part(note_array: Union[np.ndarray, list], part_id="", divs: int = None, assign_note_ids: bool = True,
                       ensurelist: bool = False, estimate_key: bool = False) -> Union[Union[Part, PartGroup], list]:
    """
    A generic function to transform an enriched note_array to part.

    The function can be used for many different occasions, i.e. load_score_midi, part_from_match, part_from_graph, etc.
    This function requires a note array that contains time signatures and key signatures(optional - can also estimate it automatically).
    Note array should contain the following fields:
    - onset_div or onset_beat
    - duration_div or duration_beat
    - pitch
    - ts_beats
    - ts_beat_type
    - key_mode(optional)
    - key_fifths(optional)
    - id(required but can also be empty)

    Parameters
    ----------
    note_array : structure array or list of structured arrays
    divs : int (optional)
        Necessary Provided divs for midi import.
    assign_note_ids : bool (optional)
        Assign note_ids
    ensurelist: bool (optional)
        ensure that output part is a list
    estimate_key: bool (optional)

    Returns
    -------
    part : list or Part or PartGroup
        Maybe should return score
    """
    if isinstance(note_array, list):
        parts = [
            note_array_to_part(note_array=x, part_id=str(i), assign_note_ids=assign_note_ids, ensurelist=ensurelist) for
            i, x in enumerate(note_array)]
        return parts

    if not isinstance(note_array, np.ndarray):
        raise TypeError("The note array does not have the correct format.")
    if len(note_array) == 0:
        raise ValueError("The note array is empty.")

    if assign_note_ids or np.all(note_array["id"] == note_array["id"][0]):
        note_ids = ["{}n{:4d}".format(part_id, i) for i in range(len(note_array))]
        note_array["id"] = np.array(note_ids)

    dtypes = note_array.dtype.names
    if not "ts_beats" in dtypes:
        raise AttributeError("The note array does not contain a time signature.")
    if all([x in dtypes for x in ["onset_div", "pitch", "duration_div", "onset_beat", "duration_beat"]]):
        pass
    elif all([x in dtypes for x in ["onset_beat", "duration_beat", "pitch", "global_score_time"]]):
        x = np.unique(note_array["onset_beat"])
        renorm_onset_beat = False
        if x.max() > note_array["ts_beats"].max():
            pass
        else:
            for unique_ons in x:
                tmp = note_array[note_array["onset_beat"] == unique_ons]
                if not np.all(tmp["global_score_time"] == tmp[0]["global_score_time"]):
                    renorm_onset_beat = True
                    break

        if renorm_onset_beat:
            tmp = note_array[0]["onset_beat"]
            tmp_ts = note_array[0]["ts_beats"]
            for idx in range(1, len(note_array)):
                if note_array[idx]["onset_beat"] < tmp:
                    tmp_ts = tmp_ts + note_array[idx]["ts_beats"] if note_array[idx][
                                                                         "onset_beat"] + tmp_ts < tmp else tmp_ts
                    tmp = note_array[idx]["onset_beat"] + tmp_ts
                    note_array[idx]["onset_beat"] = tmp
        note_array = create_divs_from_beats(note_array)
    elif all([x in dtypes for x in ["onset_beat", "duration_beat", "pitch"]]):
        note_array = create_divs_from_beats(note_array)
    elif all([x in dtypes for x in ["onset_div", "pitch", "duration_div"]]):
        pass
    else:
        raise AttributeError("The note array does not include the necessary fields.")

    if "voice" in dtypes:
        estimate_voice_info = False
        part_voice_list = note_array["voice"]
    else:
        estimate_voice_info = True
        part_voice_list = np.full(len(note_array), np.inf)

    if not all(x in dtypes for x in ['step', 'alter', 'octave']):
        warnings.warn("pitch spelling")
        spelling_global = analysis.estimate_spelling(note_array)
        note_array = rfn.merge_arrays((note_array, spelling_global), flatten=True)

    if estimate_voice_info:
        warnings.warn("voice estimation", stacklevel=2)
        # TODO: deal with zero duration notes in note_array.
        # Zero duration notes are currently deleted
        estimated_voices = analysis.estimate_voices(note_array)
        assert len(part_voice_list) == len(estimated_voices)
        for part_voice, voice_est in zip(part_voice_list, estimated_voices):
            if part_voice == np.inf:
                part_voice = voice_est
        note_array = rfn.merge_arrays((note_array, np.array(estimated_voices, dtype=[("voice", int)])), flatten=True)

    if estimate_key or ('ks_fifths' not in dtypes and 'ks_mode' not in dtypes):
        warnings.warn("key estimation", stacklevel=2)
        k_name = analysis.estimate_key(note_array)
        global_key_sigs = [[0, k_name]]
    else:
        global_key_sigs = [[0, fifths_mode_to_key_name(note_array[0]["ks_fifths"], note_array[0]["ks_mode"])]]
        for n in note_array:
            if n["ts_beats"] != global_key_sigs[-1][1] or n["ts_beat_type"] != global_key_sigs[-1][2]:
                global_key_sigs.append([n["onset_div"], fifths_mode_to_key_name(n["ks_fifths"], n["ks_mode"])])
    global_key_sigs = np.array(global_key_sigs)
    # for convenience we add the end times for each time signature
    ks_end_times = np.r_[global_key_sigs[1:, 0], np.max(note_array["onset_div"]+note_array["duration_div"])]
    global_key_sigs = np.column_stack((global_key_sigs, ks_end_times))

    if divs is None:
        divs = int((note_array[0]["duration_div"] / note_array[0]["duration_beat"])*(note_array[0]["ts_beat_type"]/4))
    part = create_part(
        ticks=divs,
        note_array=note_array,
        key_sigs=global_key_sigs,
        part_id=part_id,
        part_name=None,
    )
    return part