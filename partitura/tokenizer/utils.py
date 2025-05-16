
def midi_to_step_octave_alter(midi_pitch):
    """
    Convert MIDI pitch to (step, octave, alter).
    Assumes equal temperament and C major.

    Returns:
        step: str, one of {"C", "D", "E", "F", "G", "A", "B"}
        octave: int
        alter: int or None
    """
    step_names = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
    natural_map = {
        "C": ("C", 0),
        "C#": ("C", 1),
        "D": ("D", 0),
        "D#": ("D", 1),
        "E": ("E", 0),
        "F": ("F", 0),
        "F#": ("F", 1),
        "G": ("G", 0),
        "G#": ("G", 1),
        "A": ("A", 0),
        "A#": ("A", 1),
        "B": ("B", 0),
    }

    step_name = step_names[midi_pitch % 12]
    octave = (midi_pitch // 12) - 1
    step, alter = natural_map[step_name]
    return step, octave, alter