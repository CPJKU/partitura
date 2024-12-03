import re
import partitura.score as spt
import partitura.io as sptio
import os.path as osp
import urllib.request
from partitura.utils.music import key_name_to_fifths_mode


def load_rntxt(path: spt.Path, part=None, return_part=False):
    if sptio.is_url(path):
        data = load_data_from_url(path)
        lines = data.split("\n")
    else:
        if not osp.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r") as f:
            lines = f.readlines()
            assert validate_rntxt(lines)

    # remove empty lines
    lines = [line for line in lines if line.strip()]

    parser = RntxtParser(part)
    parser.parse(lines)
    if return_part or part is None:
        return parser.part
    return


def validate_rntxt(lines):
    # TODO: Implement
    return True


def load_data_from_url(url: str):
    with urllib.request.urlopen(url) as response:
        data = response.read().decode()
    return data


class RntxtParser:
    """
    A parser for RNtxt format to a partitura Part.

    For full specification of the format visit:
    https://github.com/MarkGotham/When-in-Rome/blob/master/syntax.md
    """
    def __init__(self, score=None):
        # Initialize parser state
        self.part = spt.Part(id="rn", part_name="Rn", part_abbreviation="rnp")
        self.current_measure = None
        self.current_position = 0
        self.measure_beat_position = 1
        self.current_time_signature = spt.TimeSignature(4, 4)
        self.time_signature_style = 'Normal'  # 'Normal', 'Slow', 'Fast'
        self.key = 'C'
        self.pedal = None
        self.metadata = {}
        self.measures = {}
        # If a score is provided, copy relevant information
        if score is not None:
            self._initialize_from_score(score)
        else:
            # Add default staff
            self.part.add(spt.Staff(number=1, lines=5), 0)

    def _initialize_from_score(self, score):
        # Copy measures, time signatures, and key signatures from the reference score
        self.ref_part = score.parts[0]
        for measure in self.ref_part.measures:
            self.part.add(measure, measure.start.t, measure.end.t)
        for time_sig in self.ref_part.time_sigs:
            self.part.add(time_sig, time_sig.start.t)
        for key_sig in self.ref_part.key_sigs:
            self.part.add(key_sig, key_sig.start.t)
        self.measures = {m.number: m for m in self.part.measures}

    def parse(self, lines):
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                if ':' in line:
                    keyword = line.split(':', 1)[0].strip()
                    if keyword in ('Composer', 'Title', 'Analyst'):
                        self._handle_metadata(line)
                    elif keyword == 'Note':
                        self._handle_note_line(line)
                    elif keyword == 'Time Signature':
                        self._handle_time_signature(line)
                    elif keyword == 'Pedal':
                        self._handle_pedal(line)
                    else:
                        self._handle_line(line)
                else:
                    self._handle_line(line)
            except Exception as e:
                print(f"Error parsing line {line_num}: {line}")
                print(e)
        self._calculate_ending_times()

    def _handle_metadata(self, line):
        key, value = line.split(':', 1)
        self.metadata[key.strip()] = value.strip()

    def _handle_note_line(self, line):
        # Notes can be stored or logged as needed
        pass

    def _handle_pedal(self, line):
        # Parse pedal information
        pass

    def _handle_line(self, line):
        if re.match(r'm\d+(-\d+)?\s*=', line):
            self._handle_repeat(line)
        elif line.startswith('m'):
            self._handle_measure(line)
        else:
            raise ValueError(f"Unknown line format: {line}")

    def _handle_repeat(self, line):
        # Implement repeat logic
        pass

    def _handle_measure(self, line):
        elements = line.strip().split()
        measure_info = elements[0]
        measure_match = re.match(r'm(\d+)(?:-(\d+))?', measure_info)
        if not measure_match:
            raise ValueError(f"Invalid measure number: {measure_info}")
        measure_number = int(measure_match.group(1))
        if measure_number not in self.measures:
            # Check if previous measure is there
            if measure_number - 1 in self.measures:
                previous_measure_start = self.measures[measure_number - 1].start.t
                # get the current time signature
                current_time_signature_beats = self.current_time_signature.beats
                self.current_position = self.part.beat_map(
                    self.part.inv_beat_map(previous_measure_start) + current_time_signature_beats)
            self.current_measure = spt.Measure(number=measure_number)
            self.measures[measure_number] = self.current_measure
            self.part.add(self.current_measure, self.current_position)
        else:
            self.current_measure = self.measures[measure_number]
        self.current_position = self.current_measure.start.t
        self.measure_beat_position = 1
        for element in elements[1:]:
            self._handle_element(element)

    def _handle_element(self, element):
        if element.startswith('b'):
            beat_match = re.match(r'b(\d+(\.\d+)?)', element)
            if beat_match:
                self.measure_beat_position = float(beat_match.group(1))
                self._update_current_position()
            else:
                raise ValueError(f"Invalid beat format: {element}")
        elif re.match(r'.*:', element):
            self._handle_key(element)
        elif all(c in "|:" for c in element):
            self._handle_barline(element)
        else:
            self._handle_roman_numeral(element)

    def _update_current_position(self):
        self.current_position = self.part.beat_map(self.part.inv_beat_map(self.current_measure.start.t) + self.measure_beat_position)

    def _get_beat_duration(self):
        # Calculate beat duration based on the time signature and style
        nom, denom = self.current_time_signature.beats, self.current_time_signature.beat_type
        quarter_note_duration = 1  # Assuming a quarter note duration of 1
        beat_duration = (4 / denom) * quarter_note_duration
        if self.time_signature_style == 'Fast' and nom % 3 == 0 and denom == 8:
            beat_duration *= 3  # Compound meter counted in dotted quarters
        elif self.time_signature_style == 'Slow' and nom % 3 == 0 and denom == 8:
            beat_duration /= 3  # Compound meter counted in eighth notes
        return beat_duration

    def _handle_time_signature(self, line):
        time_signature = line.split(':', 1)[1].strip()
        style = 'Normal'
        if 'Slow' in time_signature:
            style = 'Slow'
            time_signature = time_signature.replace('Slow', '').strip()
        elif 'Fast' in time_signature:
            style = 'Fast'
            time_signature = time_signature.replace('Fast', '').strip()
        if time_signature == 'C':
            nom, denom = 4, 4
        elif time_signature == 'Cut':
            nom, denom = 2, 2
        else:
            nom, denom = map(int, time_signature.split('/'))
        self.current_time_signature = spt.TimeSignature(nom, denom)
        self.time_signature_style = style
        self.part.add(self.current_time_signature, self.current_position)

    def _handle_barline(self, element):
        # Implement barline handling if needed
        pass

    def _handle_roman_numeral(self, element):
        try:
            rn = spt.RomanNumeral(text=element, local_key=self.key)
            self.part.add(rn, self.current_position)
        except Exception as e:
            raise ValueError(f"Error parsing Roman numeral '{element}': {e}")

    def _handle_key(self, element):
        match = re.match(r'([A-Ga-g])([#b]*):', element)
        if not match:
            raise ValueError(f"Invalid key signature: {element}")
        note, accidental = match.groups()
        mode = 'minor' if note.islower() else 'major'
        key_name = note.upper() + accidental
        key_str = f"{key_name}{('m' if mode == 'minor' else '')}"
        fifths, mode = key_name_to_fifths_mode(key_str)
        ks = spt.KeySignature(fifths=fifths, mode=mode)
        self.key = element.strip(":")
        self.part.add(ks, self.current_position)

    def _calculate_ending_times(self):
        romans = sorted(self.part.iter_all(spt.RomanNumeral), key=lambda rn: rn.start.t)
        for i, rn in enumerate(romans[:-1]):
            rn.end = spt.TimePoint(t=romans[i + 1].start.t)
        if romans:
            last_rn = romans[-1]
            last_rn.end = self.part.end_time or (last_rn.start.t + 1)




