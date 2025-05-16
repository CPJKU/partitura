from fractions import Fraction
from math import lcm, isqrt
from collections import defaultdict
import numpy as np

from ..utils.globals import REGULAR_NUM_DENOM, VALID_TIME_SIGNATURES, PROGRAM_INSTRUMENT_MAP
from ..score import Part, Note, TimePoint
from .utils import midi_to_step_octave_alter

class REMITokenizer:
    """
    REMI-style Tokenizer for symbolic music encoding.

    Attributes:
        config (dict): Configuration dictionary for tokenizer options.
        vocabulary (dict): Mapping from event string to index.
        inverse_vocabulary (dict): Mapping from index to event string.
        events (list): List of tokenized events in string format.
        tune_in_index (list): List of tokenized events in index format.
    """

    def __init__(self, config):
        """
        Initialize REMITokenizer with user-defined config.

        Args:
            config (dict): Configuration settings for tokenizer.
        """
        # 1. Validate config is a dictionary
        assert isinstance(config, dict), "Config must be a dictionary."

        # 2. Validate required keys
        required_keys = ["position_resolution"]
        for key in required_keys:
            assert key in config, f"Missing required config key: '{key}'"

        # 3. Validate position_resolution format
        pos_res = config["position_resolution"]
        assert isinstance(pos_res, (tuple, list)), "'position_resolution' must be a tuple or list."
        assert all(isinstance(v, int) and v > 0 for v in pos_res), \
            "'position_resolution' values must be positive integers."
        assert len(pos_res) >= 1, "'position_resolution' must have at least one resolution value."

        # 4. Validate pitch_range format
        assert "pitch_range" in config, "Missing required config key: 'pitch_range'"
        assert isinstance(config["pitch_range"], (tuple, list)) and len(config["pitch_range"]) == 2, \
            "'pitch_range' must be a tuple or list of two integers."

        # 5. Warn against redundant divisors (optional but helpful)
        if any(x % y == 0 for i, x in enumerate(pos_res) for j, y in enumerate(pos_res) if i != j):
            print("[Warning] Some resolution values are multiples of each other — this may be redundant.")

        # 6. Optional config: max_duration, max_denominator
        if "max_duration" in config:
            assert isinstance(config["max_duration"], int) and config["max_duration"] > 0, \
                "'max_duration' must be a positive integer."

        if "max_denominator" in config:
            assert isinstance(config["max_denominator"], int) and config["max_denominator"] >= 1, \
                "'max_denominator' must be a positive integer (>=1)."

        # 7. Optional bool flags
        for flag in ["use_velocity", "use_chords", "use_tempos", "is_multi_instrument"]:
            if flag in config:
                assert isinstance(config[flag], bool), f"'{flag}' must be a boolean."

        # Passed all checks — assign
        self.config = config
        self.index_to_token_string = {} # index -> token
        self.events = []
        self.tune_in_index = []
        
        # create vocabulary from config
        self.index_to_token_string = self.create_vocab_from_config(config)
        self.token_string_to_index = {v: int(k) for k, v in self.index_to_token_string.items()}


    def generate_position_tokens(self, position_resolution, max_measure_in_quarter=8):
        """
        Generate position tokens using union of resolutions like (4,3)
        Position names like: Position_0, Position_1/4, Position_1/3, Position_5/4, etc.

        Args:
            position_resolution (tuple): e.g., (4, 3)
            max_measure_in_quarter (int): e.g., 8 (for 8/4)

        Returns:
            list: Position tokens
        """
        # Calculate unique positions within a single quarter note
        positions_within_quarter = set()

        for res in position_resolution:
            for i in range(1, res):
                positions_within_quarter.add(Fraction(i, res))

        positions_within_quarter = sorted(list(positions_within_quarter))

        tokens = []

        for measure in range(max_measure_in_quarter):
            base = Fraction(measure, 1)  # e.g., 1, 2, 3, ...
            tokens.append(f"Position_{measure}")  # Downbeat position

            for pos in positions_within_quarter:
                full_pos = base + pos
                # Express as numerator/denominator
                if full_pos.denominator == 1:
                    tokens.append(f"Position_{full_pos.numerator}")
                else:
                    tokens.append(f"Position_{full_pos.numerator}/{full_pos.denominator}")

        return tokens


    def is_prime(self, n):
        """Return True if n is a prime number (simple check)."""
        if n < 2:
            return False
        for i in range(2, isqrt(n)+1):
            if n % i == 0:
                return False
        return True


    def generate_pruned_duration_bins(self, position_resolution, max_duration=8, min_unit=None, max_denominator=16):
        """
        Generate duration tokens using user-defined position resolution and prune
        based on musically meaningful duration values (exclude large primes, etc.).

        Args:
            position_resolution (tuple): e.g., (4, 3)
            max_duration (int): Longest duration allowed in beats.
            min_unit (Fraction or None): Force inclusion of small duration (e.g., 1/12)
            max_denominator (int): Denominator limit to avoid overly complex fractions.

        Returns:
            list: Duration tokens like 'Duration_1/4', 'Duration_2/3', etc.
        """
        durations = set()

        # Step 1: Create raw durations from each grid
        for res in position_resolution:
            for n in range(1, max_duration * res + 1):
                frac = Fraction(n, res)
                if frac <= max_duration:
                    durations.add(frac)

        # Step 2: Always include smallest unit
        smallest = Fraction(1, lcm(*position_resolution)) if min_unit is None else Fraction(min_unit)
        durations.add(smallest)

        # Step 3: Add explicit common durations
        common_fractions = [
            Fraction(1, 4), Fraction(1, 3), Fraction(1, 2),
            Fraction(2, 3), Fraction(3, 4), Fraction(1),
            Fraction(3, 2), Fraction(2), Fraction(3),
            Fraction(4), Fraction(6), Fraction(8)
        ]
        durations.update(common_fractions)

        # Step 4: Prune
        def is_musically_valid(frac):
            if frac > max_duration:
                return False
            if frac.denominator > max_denominator:
                return False
            if self.is_prime(frac.numerator) and frac.numerator not in {2, 3, 5}:
                return False
            return True

        pruned = sorted([d for d in durations if is_musically_valid(d)])

        # Step 5: Convert to token strings
        tokens = []
        for dur in pruned:
            if dur.denominator == 1:
                tokens.append(f"Note_Duration_{dur.numerator}")
            else:
                tokens.append(f"Note_Duration_{dur.numerator}/{dur.denominator}")

        return tokens


    def create_vocab_from_config(self, config):
        vocab = {}
        idx = 0

        # Metric Tokens
        metric_tokens = ["SOS_None", "EOS_None", "Bar_None"]
        metric_tokens += [f"Bar_{ts}" for ts in VALID_TIME_SIGNATURES]

        for token in metric_tokens:
            vocab[str(idx)] = token
            idx += 1

        # Position Tokens
        position_tokens = self.generate_position_tokens(config["position_resolution"], max_measure_in_quarter=8)

        for token in position_tokens:
            vocab[str(idx)] = token
            idx += 1

        # Duration Tokens
        duration_tokens = self.generate_pruned_duration_bins(config["position_resolution"], max_duration=8, min_unit=None)

        for token in duration_tokens:
            vocab[str(idx)] = token
            idx += 1

        # Pitch Tokens
        low, high = config["pitch_range"]
        assert low >= 0 and high <= 127 and low < high, "Invalid MIDI pitch range."

        pitch_tokens = [f"Note_Pitch_{p}" for p in range(low, high + 1)]

        for token in pitch_tokens:
            vocab[str(idx)] = token
            idx += 1

        return vocab


    def get_position_grid_ticks(self, position_resolution, measure_quarters, tpq):
        """
        Create a sorted list of absolute tick positions for a measure
        based on given position resolution and TPQ (ticks per quarter note).

        Returns:
            grid_map: dict mapping absolute tick → fractional position within measure
        """
        grid_fractions = set()

        # For each quarter in the measure, apply all position resolutions
        for beat_idx in range(int(measure_quarters)):
            for res in position_resolution:
                for i in range(res):
                    frac = Fraction(i, res)
                    full_frac = Fraction(beat_idx) + frac
                    grid_fractions.add(full_frac)

        # Also add the downbeat at the end of the measure
        grid_fractions.add(Fraction(measure_quarters))

        grid_map = {}
        for frac in sorted(grid_fractions):
            tick = round(frac * tpq)
            grid_map[tick] = frac

        return grid_map


    def prune_notes_keep_best(self, notes):
        """
        From a list of notes quantized to the same position,
        retain only the best one based on:
        1. Longest duration
        2. Highest pitch (if tie)
        """
        if not notes:
            return []

        # Sort by: (longest duration → highest pitch)
        sorted_notes = sorted(
            notes,
            key=lambda x: (x["duration"], x["pitch"]),
            reverse=True
        )

        return [sorted_notes[0]]  # Keep only the best note


    def encode_to_event(self, part, config):
        """
        Encode a partitura Score object into REMI token sequence.

        Args:
            part (partitura.score.Score): Score object to tokenize.

        Returns:
            self: With updated self.events and self.tune_in_index
        """
        events = [{'name': 'SOS', 'value': None}]

        tpq = part.quarter_durations()[0][1]
        note_array = part.note_array()
        resolution = config["position_resolution"]

        # Time signatures by tick
        time_signatures_dict = {}
        for ts in part.time_sigs:
            time_signatures_dict[ts.start.t] = (ts.beats, ts.beat_type)

        # Duration grid setup
        duration_grid = self.generate_pruned_duration_bins(
            config["position_resolution"],
            max_duration=config.get("max_duration", 8),
            max_denominator=config.get("max_denominator", 16)
        )
        duration_map = {
            token: Fraction(token.replace("Note_Duration_", "")) for token in duration_grid
        }
        duration_list = list(duration_map.values())

        for measure in part.measures:
            note_idx = np.where((note_array['onset_div'] >= measure.start.t) &
                                (note_array['onset_div'] < measure.end.t))[0]
            note_array_seg = note_array[note_idx]

            # Current time signature
            for ts_start, ts in sorted(time_signatures_dict.items()):
                if ts_start <= measure.start.t:
                    current_time_signature = ts
                else:
                    break

            expected_quarter = (current_time_signature[0] * 4 / current_time_signature[1])
            actual_quarter = measure.duration / tpq

            if expected_quarter != actual_quarter:
                beats_adjusted = actual_quarter * current_time_signature[1] / 4
                time_signature = (int(beats_adjusted), current_time_signature[1])
            else:
                time_signature = current_time_signature

            if time_signature not in REGULAR_NUM_DENOM:
                continue
            
            time_signature_string = f"time_signature_{time_signature[0]}/{time_signature[1]}"
            events.append({'name': 'Bar', 'value': time_signature_string})

            # Position quantization grid
            grid_map = self.get_position_grid_ticks(resolution, actual_quarter, tpq)
            grid_ticks = np.array(list(grid_map.keys()))

            position_to_notes = defaultdict(list)

            for note in note_array_seg:
                note_onset_tick = note["onset_div"] - measure.start.t
                closest_tick_idx = np.argmin(np.abs(grid_ticks - note_onset_tick))
                closest_tick = grid_ticks[closest_tick_idx]
                closest_frac = grid_map[closest_tick]

                if float(closest_frac) > 8.0:
                    continue

                duration_frac = Fraction(float(note["duration_quarter"])).limit_denominator(32)
                closest_d_idx = np.argmin([abs(duration_frac - d) for d in duration_list])
                closest_d = duration_list[closest_d_idx]

                position_to_notes[closest_frac].append({
                    "pitch": int(note["pitch"]),
                    "duration": closest_d,
                    "original_onset": note["onset_quarter"]
                })

            for frac_position in sorted(position_to_notes.keys()):
                if frac_position.denominator == 1:
                    position_token = f"{frac_position.numerator}"
                else:
                    position_token = f"{frac_position.numerator}/{frac_position.denominator}"

                events.append({'name': 'Position', 'value': position_token})

                for note in self.prune_notes_keep_best(position_to_notes[frac_position]):
                    pitch = note["pitch"]
                    duration = note["duration"]
                    if duration.denominator == 1:
                        duration_token = f"{duration.numerator}"
                    else:
                        duration_token = f"{duration.numerator}/{duration.denominator}"

                    events.append({'name': 'Note_Pitch', 'value': pitch})
                    events.append({'name': 'Note_Duration', 'value': duration_token})

        # End-of-score tokens
        events.append({'name': 'Bar', 'value': None})
        events.append({'name': 'EOS', 'value': None})
        
        return events


    def encode_to_index(self, events):
        """
        Convert token strings to indices using the vocabulary.
        """
        tune_in_index = []
        for event in events:
            token_str = f"{event['name']}_{event['value']}" if event['value'] is not None else f"{event['name']}_None"
            if token_str in self.token_string_to_index:
                tune_in_index.append(self.token_string_to_index[token_str])
            else:
                raise KeyError(f"Token '{token_str}' not found in vocabulary.")
        return tune_in_index


    def decode_from_event(self, events, config):
        """
        Decode a list of REMI-style events back into a partitura Part object.

        Args:
            events (List[Dict]): List of {"name": ..., "value": ...} events.
            config (Dict): Tokenizer config with position_resolution etc.

        Returns:
            partitura.score.Part: A valid Part object reconstructed from REMI events.
        """
        part = Part(id="decoded_part")
        tpq = 480  # ticks per quarter note
        part.set_quarter_duration(0, tpq)
        current_tick_offset = 0  # start of current measure
        current_position_tick = 0
        current_time_signature = (4, 4)

        notes = []
        i = 0

        while i < len(events):
            evt = events[i]
            name, value = evt["name"], evt["value"]

            if name == "Bar":
                if value is not None: # e.g., "'time_signature_4/4'"
                    current_time_signature = value.split("_")[-1].split("/")
                # Advance offset for next bar
                measure_quarters = int(current_time_signature[0]) * 4 / int(current_time_signature[1])
                current_tick_offset += int(measure_quarters * tpq)
                i += 1

            elif name == "Position":
                pos_str = str(value)
                if "/" in pos_str:
                    numerator, denominator = map(int, pos_str.split('/'))
                    frac = Fraction(numerator, denominator)
                else:
                    frac = Fraction(int(pos_str), 1)
                current_position_tick = current_tick_offset + round(frac * tpq)
                i += 1

            elif name == "Note_Pitch" and i + 1 < len(events) and events[i+1]["name"] == "Note_Duration":
                pitch = int(value)
                dur_str = str(events[i+1]["value"])
                if "/" in dur_str:
                    numerator, denominator = map(int, dur_str.split('/'))
                    duration_frac = Fraction(numerator, denominator)
                else:
                    duration_frac = Fraction(int(dur_str), 1)
                duration_ticks = round(duration_frac * tpq)

                step, octave, alter = midi_to_step_octave_alter(pitch)

                step, octave, alter = midi_to_step_octave_alter(pitch)

                # Create a new Note (step/octave/alter already parsed)
                note = Note(
                    step=step,
                    octave=octave,
                    alter=alter,
                    voice=0
                )

                # Link to timeline using get_or_add_point
                start_tp = part.get_or_add_point(current_position_tick)
                end_tp = part.get_or_add_point(current_position_tick + duration_ticks)

                note.start = start_tp
                note.end = end_tp

                # Now add to the Part
                part.add(note, start=start_tp.t, end=end_tp.t)
                i += 2

            else:
                i += 1

        return part


    def train(self, midi_folder_path):
        """
        Train a customized vocabulary from user-provided MIDI files.

        Args:
            midi_folder_path (str or Path): Path to MIDI files.

        Returns:
            self: With updated self.index_to_token_string and self.token_string_to_index
        """
        # Step 1: Parse all MIDI files
        
        # Step 2: Extract events from all scores
        
        # Step 3: Build vocabulary (event -> index)