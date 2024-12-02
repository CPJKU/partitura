import warnings
import numpy as np
from math import ceil
import partitura.score as spt
from partitura.score import process_local_key
from partitura.utils.music import estimate_symbolic_duration

try:
    import pandas as pd
except ImportError:
    pd = None


def read_note_tsv(note_tsv_path, metadata=None):
    data = pd.read_csv(note_tsv_path, sep="\t")
    # Hack for empty values in quarterbeats, to investigate.
    # (It happens with voltas when the second volta has a different number of measures)
    if not np.all(data["quarterbeats"].isna() == False):
        data = data[~data["quarterbeats"].isna()]
    data["quarterbeats"] = (
        data["quarterbeats"].apply(eval)
        if data.dtypes["quarterbeats"] == str or data.dtypes["quarterbeats"] == object
        else data["quarterbeats"]
    )
    unique_durations = data["duration"].unique()
    denominators = [int(qb.split("/")[1]) for qb in unique_durations if "/" in qb]
    # transform quarter_beats to quarter_divs
    qdivs = np.lcm.reduce(denominators) if len(denominators) > 0 else 4
    quarter_durations = data["duration_qb"]
    duration_div = np.array([ceil(qd * qdivs) for qd in quarter_durations])
    onset_div = np.array([ceil(qd * qdivs) for qd in data["quarterbeats"]])
    data["alter"] = data["name"].str.count("#") - data["name"].str.count("b")
    data["step"] = data["name"].apply(lambda x: x[0])
    data["onset_div"] = onset_div
    data["duration_div"] = duration_div
    data["pitch"] = data["midi"]
    grace_mask = (
        ~data["gracenote"].isna().to_numpy()
        if "gracenote" in data.columns
        else np.zeros(len(data), dtype=bool)
    )
    data["id"] = np.arange(len(data))
    # Rewrite Voices for correct export
    # taking the maximum voice number for the entire staff, and having the second staff starting from that number.
    staffs = data["staff"].unique()
    re_index_voice_value = 0
    for staff in staffs:
        staff_mask = data["staff"] == staff
        # add re_index_voice_value to the voice values of the staff
        data.loc[staff_mask, "voice"] += re_index_voice_value
        # update re_index_voice_value
        re_index_voice_value = data.loc[staff_mask, "voice"].max()

    note_array = data[
        [
            "onset_div",
            "duration_div",
            "pitch",
            "step",
            "alter",
            "octave",
            "id",
            "staff",
            "voice",
        ]
    ].to_records(index=False)
    part = spt.Part("P0", "Metadata", quarter_duration=qdivs)

    # Add notes and grace notes
    for n_idx, note in enumerate(note_array):
        if grace_mask[n_idx]:
            # verify that staff and voice are the same for the grace note and the main note
            note_el = spt.GraceNote(
                grace_type="grace",
                id="n-{}".format(note["id"]),
                step=note["step"],
                octave=note["octave"],
                alter=note["alter"],
                staff=note["staff"],
                voice=note["voice"],
                symbolic_duration={"type": "eighth"},
            )

            if grace_mask[n_idx - 1]:
                prev_note = note_array[n_idx - 1]
                for note_prev in part.iter_all(
                    spt.GraceNote, note["onset_div"], note["onset_div"] + 1
                ):
                    if note_prev.id == "n-{}".format(prev_note["id"]):
                        note_el.grace_prev = note_prev
                        note_prev.grace_next = note_el
                        break
        else:
            symbolic_duration = estimate_symbolic_duration(note["duration_div"], qdivs)
            note_el = spt.Note(
                id="n-{}".format(note["id"]),
                step=note["step"],
                octave=note["octave"],
                alter=note["alter"],
                staff=note["staff"],
                voice=note["voice"],
                symbolic_duration=symbolic_duration,
            )
        part.add(
            note_el,
            start=note["onset_div"],
            end=(note["onset_div"] + note["duration_div"]),
        )

    # Curate grace notes
    grace_note_idxs = np.where(grace_mask)[0]
    for i, grace_el in enumerate(part.iter_all(spt.GraceNote)):
        grace_idx = grace_note_idxs[i]
        note = note_array[grace_idx]
        # Find the next note in the same staff and voice
        if not grace_mask[grace_idx + 1]:
            i = 1
            next_note = note_array[grace_idx + i]
            while (
                note["staff"] != next_note["staff"]
                or note["voice"] != next_note["voice"]
            ):
                i += 1
                next_note = note_array[grace_idx + i]
                if i > 10:
                    warnings.warn(
                        "Grace note ignored, no matching main note found within 10 notes."
                    )
                    break
            assert (
                note["staff"] == next_note["staff"]
            ), "Grace note and main note must be in the same staff"
            assert (
                note["voice"] == next_note["voice"]
            ), "Grace note and main note must be in the same voice"
            assert (
                note["onset_div"] == next_note["onset_div"]
            ), "Grace note and main note must have the same onset"
            for note in part.iter_all(
                spt.Note, note["onset_div"], note["onset_div"] + 1
            ):
                if note.id == "n-{}".format(next_note["id"]):
                    grace_el.grace_next = note
                    break

    # Find time signatures
    time_signatures_changes = data["timesig"][
        data["timesig"].shift(1) != data["timesig"]
    ].index
    time_signatures = data["timesig"][time_signatures_changes]
    start_divs = np.array(
        [int(qd * qdivs) for qd in data["quarterbeats"][time_signatures_changes]]
    )
    end_of_piece = (note_array["onset_div"] + note_array["duration_div"]).max()
    end_divs = np.r_[start_divs[1:], end_of_piece]
    for ts, start, end in zip(time_signatures, start_divs, end_divs):
        part.add(
            spt.TimeSignature(
                beats=int(ts.split("/")[0]), beat_type=int(ts.split("/")[1])
            ),
            start=start,
            end=end,
        )

    # Add default clefs for piano pieces (Naive)
    part.add(spt.Clef(staff=1, sign="G", line=2, octave_change=0), start=0)
    part.add(spt.Clef(staff=2, sign="F", line=4, octave_change=0), start=0)

    # Add Ties
    tied_note_mask = data["tied"] == 1
    for tied_note in note_array[tied_note_mask]:
        for note in part.iter_all(
            spt.Note, tied_note["onset_div"], tied_note["onset_div"] + 1
        ):
            if note.id == "n-{}".format(tied_note["id"]):
                found_next = False
                for note_next in part.iter_all(
                    spt.Note, note.end.t, note.end.t + 1, mode="starting"
                ):
                    condition = (
                        note_next.alter == note.alter
                        and note_next.step == note.step
                        and note_next.octave == note.octave
                        and note.voice == note_next.voice
                        and note.staff == note_next.staff
                    )
                    if condition:
                        note.tie_next = note_next
                        note_next.tie_prev = note
                        found_next = condition
                        break
                if not found_next:
                    warnings.warn("Opening tie, but no matching note found.")

    return part


def read_measure_tsv(measure_tsv_path, part):
    qdivs = part._quarter_durations[0]
    data = pd.read_csv(measure_tsv_path, sep="\t")
    # Hack for empty values in quarterbeats, to investigate.
    # (It happens with voltas when the second volta has a different number of measures)
    if not np.all(data["quarterbeats"].isna() == False):
        data = data[~data["quarterbeats"].isna()]
    data["quarterbeats"] = (
        data["quarterbeats"].apply(eval)
        if data.dtypes["quarterbeats"] == str or data.dtypes["quarterbeats"] == object
        else data["quarterbeats"]
    )
    data["onset_div"] = np.array([int(qd * qdivs) for qd in data["quarterbeats"]])
    data["duration_div"] = np.array([int(qd * qdivs) for qd in data["duration_qb"]])
    # Get first index
    repeat_index, _ = next(data.iterrows())

    for idx, row in data.iterrows():
        part.add(
            spt.Measure(number=row["mc"], name=row["mn"]),
            start=row["onset_div"],
            end=row["onset_div"] + row["duration_div"],
        )

        if row["repeats"] == "start":
            repeat_index = idx
        elif row["repeats"] == "end":
            # Find the previous repeat start
            start_times = data.iloc[repeat_index]["onset_div"]
            part.add(spt.Repeat(), start=start_times, end=row["onset_div"])

    part.add(spt.Fine(), start=part.last_point.t)
    return


def read_harmony_tsv(beat_tsv_path, part):
    qdivs = part._quarter_durations[0]
    data = pd.read_csv(beat_tsv_path, sep="\t")
    # Hack for empty values in quarterbeats, to investigate.
    # (It happens with voltas when the second volta has a different number of measures)
    if not np.all(data["quarterbeats"].isna() == False):
        data = data[~data["quarterbeats"].isna()]
    data["quarterbeats"] = (
        data["quarterbeats"].apply(eval)
        if data.dtypes["quarterbeats"] == str or data.dtypes["quarterbeats"] == object
        else data["quarterbeats"]
    )
    data["onset_div"] = np.array([int(qd * qdivs) for qd in data["quarterbeats"]])
    data["duration_div"] = np.array([int(qd * qdivs) for qd in data["duration_qb"]])
    is_na_cad = data["cadence"].isna()
    is_na_roman = data["chord"].isna()
    # Find Phrase Starts where data["phraseend"] == "{"
    for idx, row in data[~is_na_roman].iterrows():
        # row["chord_type"] contains the quality of the chord but it is encoded differently than for other formats
        # and datasets. For example, a minor chord is encoded as "m" instead of "min" or "minor"
        # Therefore we do not add the quality to the RomanNumeral object. Then it is extracted from the text.
        # Local key is in relation to the global key.
        if "/" in row["localkey"]:
            # if the local key has a secondary degree (e.g. "V/IV") we need to process it differently
            inter_key = process_local_key(
                row["localkey"].split("/")[-1], row["globalkey"]
            )
            local_key = process_local_key(row["localkey"].split("/")[0], inter_key)
        else:
            local_key = process_local_key(row["localkey"], row["globalkey"])

        part.add(
            spt.RomanNumeral(
                text=row["chord"],
                local_key=local_key,
                # quality=row["chord_type"],
            ),
            start=row["onset_div"],
            end=row["onset_div"] + row["duration_div"],
        )

    for idx, row in data[~is_na_cad].iterrows():
        if "/" in row["localkey"]:
            # if the local key has a secondary degree (e.g. "V/IV") we need to process it differently
            inter_key = process_local_key(
                row["localkey"].split("/")[-1], row["globalkey"]
            )
            local_key = process_local_key(row["localkey"].split("/")[0], inter_key)
        else:
            local_key = process_local_key(row["localkey"], row["globalkey"])
        # key_step = re.search(r"[a-gA-G]", row["globalkey"]).group(0)
        # key_alter = re.search(r"[#b]", row["globalkey"]).group(0) if re.search(r"[#b]", row["globalkey"]) else ""
        # key_alter = key_alter.replace("b", "-")
        # key_alter = ALT_TO_INT[key_alter]
        # key_step, key_alter = transpose_note(key_step, key_alter, transposition_interval)
        # local_key = key_step + INT_TO_ALT[key_alter]
        part.add(
            spt.Cadence(
                text=row["cadence"],
                local_key=local_key,
            ),
            start=row["onset_div"],
            end=row["onset_div"] + row["duration_div"],
        )

    # Check if phrase information is available.
    if np.all(data["phraseend"].isna()):
        return
    # search if character "{, }" in present in values of column phraseend
    phrase_starts = data[data["phraseend"].str.contains("{") == True]
    phrase_ends = data[data["phraseend"].str.contains("}") == True]
    # Check that the number of phrase starts and ends match
    if len(phrase_starts) == len(phrase_ends):
        for start, end in zip(phrase_starts.iterrows(), phrase_ends.iterrows()):
            part.add(spt.Phrase(), start=start[1]["onset_div"], end=end[1]["onset_div"])
    else:
        # TODO: account for unfoldings and repeats.
        warnings.warn(
            "Number of phrase starts and ends do not match, skipping parsing phrases"
        )
    return


def load_dcml(
    note_tsv_path, measure_tsv_path=None, harmony_tsv_path=None, metadata=None
):
    """
    Load a score from tsv files containing the notes, measures and harmony annotations.

    These files are provided by the DCML datasets.
    ATTENTION: This functionality requires pandas to be installed, which is not a requirement for partitura.

    Parameters
    ----------
    note_tsv_path: str
        Path to the tsv file containing the notes
    measure_tsv_path: str
        Path to the tsv file containing the measures
    harmony_tsv_path:
        Path to the tsv file containing the harmony annotations
    metadata: dict
        Metadata to add to the score. This is useful to add the composer, title, etc.

    Returns
    -------
    score: :class:`partitura.score.Score`
        A `Score` instance.

    """
    if pd is None:
        raise ImportError("This functionality requires pandas to be installed")

    part = read_note_tsv(note_tsv_path, metadata=metadata)
    if measure_tsv_path is not None:
        read_measure_tsv(measure_tsv_path, part)
    else:
        spt.add_measures(part)
    if harmony_tsv_path is not None:
        read_harmony_tsv(harmony_tsv_path, part)
    score = spt.Score([part])
    return score
