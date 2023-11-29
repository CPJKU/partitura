import numpy as np

import partitura.score as spt
try:
    import pandas as pd
except ImportError:
    pd = None


def read_note_tsv(note_tsv_path, metadata=None):
    data = pd.read_csv(note_tsv_path, sep="\t")
    unique_durations = data["duration"].unique()
    nominators = [int(qb.split("/")[0]) for qb in unique_durations]
    denominators = [int(qb.split("/")[1]) for qb in unique_durations if "/" in qb]
    # transform quarter_beats to quarter_divs
    qdivs = np.lcm.reduce(denominators)
    quarter_durations = data["duration_qb"]
    duration_div = np.array([int(qd * qdivs) for qd in quarter_durations])
    onset_div = np.array([int(qd * qdivs) for qd in data["quarterbeats"].apply(eval)])
    flats = data["name"].str.contains("b")
    sharps = data["name"].str.contains("#")
    double_sharps = data["name"].str.contains("##")
    double_flats = data["name"].str.contains("bb")
    alter = np.zeros(len(data), dtype=np.int32)
    alter[flats] = -1
    alter[sharps] = 1
    alter[double_sharps] = 2
    alter[double_flats] = -2
    data["step"] = data["name"].apply(lambda x: x[0])
    data["onset_div"] = onset_div
    data["duration_div"] = duration_div
    data["alter"] = alter
    data["pitch"] = data["midi"]
    grace_mask = ~data["gracenote"].isna()
    data["id"] = np.arange(len(data))
    note_array = data[["onset_div", "duration_div", "pitch", "step", "alter", "octave", "id", "staff", "voice"]].to_records(index=False)
    part = spt.Part("P0", "Metadata", quarter_duration=qdivs)

    # Add notes
    notes = note_array[~grace_mask]
    for note in notes:
        part.add(
            spt.Note(
                id=note["id"],
                step=note["step"],
                octave=note["octave"],
                alter=note["alter"],
                staff=note["staff"],
                voice=note["voice"]
            ), start=note["onset_div"], end=note["onset_div"]+note["duration_div"])
    # Add Grace notes
    grace_notes = note_array[grace_mask]
    for grace_note in grace_notes:
        part.add(
            spt.GraceNote(
                grace_type="grace",
                id=grace_note["id"],
                step=grace_note["step"],
                octave=grace_note["octave"],
                alter=grace_note["alter"],
                staff=grace_note["staff"],
                voice=grace_note["voice"]
            ),
            start=grace_note["onset_div"],
            end=grace_note["onset_div"]
        )

    # Find time signatures
    time_signatures_changes = data["timesig"][data["timesig"].shift(1) != data["timesig"]].index
    time_signatures = data["timesig"][time_signatures_changes]
    start_divs = np.array([int(qd * qdivs) for qd in data["quarterbeats"][time_signatures_changes]])
    end_of_piece = (note_array["onset_div"]+note_array["duration_div"]).max()
    end_divs = np.r_[start_divs[1:], end_of_piece]
    for ts, start, end in zip(time_signatures, start_divs, end_divs):
        part.add(spt.TimeSignature(beats=int(ts.split("/")[0]), beat_type=int(ts.split("/")[1])), start=start, end=end)

    # TODO: Find Ties
    tied_notes = data["tied"].dropna()

    return part


def read_measure_tsv(measure_tsv_path):
    return


def read_harmony_tsv(beat_tsv_path, part):
    qdivs = part._quarter_durations[0]
    data = pd.read_csv(beat_tsv_path, sep="\t")
    data["onset_div"] = np.array([int(qd * qdivs) for qd in data["quarterbeats"].apply(eval)])
    data["duration_div"] = np.array([int(qd * qdivs) for qd in data["duration_qb"]])
    for _, row in data.iterrows():
        part.add(spt.RomanNumeral(roman=row["chord"]), start=row["onset_div"], end=row["onset_div"]+row["duration_div"])
    return


def load_tsv(note_tsv_path, measure_tsv_path=None, harmony_tsv_path=None, metadata=None):
    part = read_note_tsv(note_tsv_path, metadata=metadata)
    if measure_tsv_path is not None:
        read_measure_tsv(measure_tsv_path, part)
    else:
        spt.add_measures(part)
    if harmony_tsv_path is not None:
        read_harmony_tsv(harmony_tsv_path, part)
    score = spt.Score([part])
    return score

