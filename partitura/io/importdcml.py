import numpy as np
from math import ceil
import partitura.score as spt
from partitura.utils.music import estimate_symbolic_duration
try:
    import pandas as pd
except ImportError:
    pd = None


def read_note_tsv(note_tsv_path, metadata=None):
    data = pd.read_csv(note_tsv_path, sep="\t")
    unique_durations = data["duration"].unique()
    denominators = [int(qb.split("/")[1]) for qb in unique_durations if "/" in qb]
    # transform quarter_beats to quarter_divs
    qdivs = np.lcm.reduce(denominators) if len(denominators) > 0 else 4
    quarter_durations = data["duration_qb"]
    duration_div = np.array([ceil(qd * qdivs) for qd in quarter_durations])
    onset_div = np.array([ceil(qd * qdivs) for qd in data["quarterbeats"].apply(eval)])
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
        symbolic_duration = estimate_symbolic_duration(note["duration_div"], qdivs)
        part.add(
            spt.Note(
                id="n-{}".format(note["id"]),
                step=note["step"],
                octave=note["octave"],
                alter=note["alter"],
                staff=note["staff"],
                voice=note["voice"],
                symbolic_duration=symbolic_duration
            ), start=note["onset_div"], end=(note["onset_div"]+note["duration_div"]))
    # Add Grace notes
    grace_note_idxs = np.where(grace_mask)[0]
    grace_note_idxs = []
    for grace_idx in grace_note_idxs:
        grace_note = note_array[grace_idx]
        grace_el = spt.GraceNote(
            grace_type="grace",
            id="n-{}".format(grace_note["id"]),
            step=grace_note["step"],
            octave=grace_note["octave"],
            alter=grace_note["alter"],
            staff=grace_note["staff"],
            voice=grace_note["voice"],
            symbolic_duration={"type": "eighth"},
        )

        if grace_mask[grace_idx-1]:
            prev_note = note_array[grace_idx-1]
            for note in part.iter_all(spt.GraceNote, grace_note["onset_div"], grace_note["onset_div"]+1):
                if note.id == "n-{}".format(prev_note["id"]):
                    grace_el.grace_prev = note
                    note.grace_next = grace_el
                    break

        part.add(
            grace_el, grace_note["onset_div"], grace_note["onset_div"]
        )

    # Find time signatures
    time_signatures_changes = data["timesig"][data["timesig"].shift(1) != data["timesig"]].index
    time_signatures = data["timesig"][time_signatures_changes]
    start_divs = np.array([int(qd * qdivs) for qd in data["quarterbeats"][time_signatures_changes]])
    end_of_piece = (note_array["onset_div"]+note_array["duration_div"]).max()
    end_divs = np.r_[start_divs[1:], end_of_piece]
    for ts, start, end in zip(time_signatures, start_divs, end_divs):
        part.add(spt.TimeSignature(beats=int(ts.split("/")[0]), beat_type=int(ts.split("/")[1])), start=start, end=end)

    # Add default clefs for piano pieces (Naive)
    part.add(spt.Clef(staff=1, sign="G", line=2, octave_change=0), start=0)
    part.add(spt.Clef(staff=2, sign="F", line=4, octave_change=0), start=0)
    # TODO: Find Ties
    tied_notes = data["tied"].dropna()

    return part


def read_measure_tsv(measure_tsv_path, part):
    qdivs = part._quarter_durations[0]
    data = pd.read_csv(measure_tsv_path, sep="\t")
    data["onset_div"] = np.array([int(qd * qdivs) for qd in data["quarterbeats"]])
    data["duration_div"] = np.array([int(qd * qdivs) for qd in data["duration_qb"]])
    repeat_index = 0

    for idx, row in data.iterrows():
        part.add(spt.Measure(), start=row["onset_div"], end=row["onset_div"]+row["duration_div"])
        # if row["repeat"] == "start":
        if row["repeats"] == "start":
            repeat_index = idx
        elif row["repeats"] == "":
            # Find the previous repeat start
            start_times = data[repeat_index]["onset_div"]
            part.add(spt.Repeat(), start=start_times, end=row["onset_div"])


def read_harmony_tsv(beat_tsv_path, part):
    qdivs = part._quarter_durations[0]
    data = pd.read_csv(beat_tsv_path, sep="\t")
    data["onset_div"] = np.array([int(qd * qdivs) for qd in data["quarterbeats"].apply(eval)])
    data["duration_div"] = np.array([int(qd * qdivs) for qd in data["duration_qb"]])
    is_na_cad = data["cadence"].isna()
    is_na_roman = data["chord"].isna()
    # Find Phrase Starts where data["phraseend"] == "{"
    for idx, row in data[~is_na_roman].iterrows():
        part.add(
            spt.RomanNumeral(text=row["chord"],
                             local_key=row["localkey"],
                             quality=row["chord_type"],
                             ), start=row["onset_div"], end=row["onset_div"]+row["duration_div"])

    for idx, row in data[~is_na_cad].iterrows():
        part.add(
            spt.Cadence(text=row["cadence"],
                        local_key=row["localkey"],
                        ), start=row["onset_div"], end=row["onset_div"]+row["duration_div"])

    phrase_starts = data[data["phraseend"] == "{"]
    phrase_ends = data[data["phraseend"] == "}"]
    # Check that the number of phrase starts and ends match
    assert len(phrase_starts) == len(phrase_ends), "Number of phrase starts and ends do not match"
    for start, end in zip(phrase_starts.iterrows(), phrase_ends.iterrows()):
        part.add(spt.Phrase(), start=start[1]["onset_div"], end=end[1]["onset_div"])
    return


def load_tsv(note_tsv_path, measure_tsv_path=None, harmony_tsv_path=None, metadata=None):
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

