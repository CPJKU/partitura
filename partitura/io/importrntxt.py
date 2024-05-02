import partitura.score as spt
import os.path as osp
import numpy as np
from urllib.parse import urlparse
import urllib.request
from partitura.utils.music import key_name_to_fifths_mode


def load_rntxt(path: spt.Path, part=None, return_part=False):

    if is_url(path):
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


def is_url(input):
    try:
        result = urlparse(input)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


class RntxtParser:
    def __init__(self, score=None):
        if score is not None:
            self.ref_part = score.parts[0]
            quarter_duration = self.ref_part._quarter_durations[0]
            ref_measures = self.ref_part.measures
        else:
            quarter_duration = 4
            ref_measures = []
        self.part = spt.Part(id="rn", part_name="Rn", part_abbreviation="rnp", quarter_duration=quarter_duration)
        # include measures
        for measure in ref_measures:
            self.part.add(measure, measure.start.t, measure.end.t)
        self.measures = {m.number: m for m in self.part.measures}
        self.current_measure = None
        self.current_position = 0
        self.measure_beat_position = 1
        self.current_voice = None
        self.current_note = None
        self.current_chord = None
        self.current_tie = None
        self.num_parsed_romans = 0
        self.key = "C"

    def parse(self, lines):
        # np_lines = np.array(lines)
        # potential_measure_lines = np.lines[np.char.startswith(np_lines, "m")]
        # for line in potential_measure_lines:
        #     self._handle_measure(line)
        for line in lines:
            if line.startswith("Time Signature:"):
                self.time_signature = line.split(":")[1].strip()
            elif line.startswith("Pedal:"):
                self.pedal = line.split(":")[1].strip()
            elif line.startswith("m"):
                self._handle_measure(line)

    def _handle_measure(self, line):
        if not self._validate_measure_line(line):
            return
        elements = line.split(" ")
        measure_number = elements[0].strip("m")
        if not measure_number.isnumeric():
            # TODO: check if it is a valid measure number or variation
            raise ValueError(f"Invalid measure number: {measure_number}")
        measure_number = int(measure_number)
        if measure_number not in self.measures.keys():
            self.current_measure = spt.Measure(number=measure_number)
            self.measures[measure_number] = self.current_measure
            self.part.add(self.current_measure, self.current_position)
        else:
            self.current_measure = self.measures[measure_number]

        self.current_position = self.current_measure.start.t
        # starts counting beats from 1
        self.measure_beat_position = 1
        for element in elements[1:]:
            self._handle_element(element)

    def _handle_element(self, element):
        # if element starts with "b" followed by a number ("float" or "int") it is a beat
        if element.startswith("b") and element[1:].replace(".", "").isnumeric():
            self.measure_beat_position = float(element[1:])
            if self.current_measure.number == 0:
                if (self.current_position == 0 and self.num_parsed_romans == 0):
                    self.current_position = 0
                else:
                    self.current_position = self.part.inv_beat_map(self.part.beat_map(self.current_position) + self.measure_beat_position - 1)
            else:
                self.current_position = self.part.inv_beat_map(self.part.beat_map(self.current_measure.start.t) + self.measure_beat_position - 1)

        # if element starts with [A-G] and it includes : it is a key
        elif element[0] in "ABCDEFG" and ":" in element:
            self._handle_key(element)
        # if element only contains "|" or ":" (and combinations) it is a barline
        elif all(c in "|:" for c in element):
            self._handle_barline(element)
        # else it is a roman numeral
        else:
            self._handle_roman_numeral(element)

    def _handle_key(self, element):
        # key is in the format "C:" or "c:" for C major or c minor
        # for alterations use "C#:" or "c#:" for C# major or c# minor
        name = element[0]
        mode = "major" if name.isupper() else "minor"
        step = name.upper()
        # handle alterations
        alter = element.count("#") - element.count("b")
        # step and alter to fifths
        fifths, mode = key_name_to_fifths_mode(element.strip(":"))
        ks = spt.KeySignature(fifths=fifths, mode=mode)
        self.key = element.strip(":")
        self.part.add(ks, self.current_position)

    def _handle_barline(self, element):
        pass

    def _handle_roman_numeral(self, element):
        element = element.strip()
        rn = spt.RomanNumeral(text=element, local_key=self.key)
        self.part.add(rn, self.current_position)
        self.num_parsed_romans += 1

    def _validate_measure_line(self, line):
        # does it have elements
        if not len(line.split(" ")) > 1:
            return False
        return True



