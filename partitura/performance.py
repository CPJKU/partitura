import numpy as np

from partitura.utils import (
    pitch_spelling_to_midi_pitch
)


class PerformedNote(object):
    """A class for representing performed MIDI notes
    """

    def __init__(self, id, midi_pitch, note_on,
                 note_off, velocity,
                 sound_off=None, step=None, octave=None,
                 alter=None, use_sound_off=False):

        self.id = id
        self.midi_pitch = midi_pitch
        self.note_on = note_on
        self.note_off = note_off
        self.sound_off = sound_off
        if self.sound_off is None:
            self.sound_off = note_off

        self.velocity = velocity
        self.step = step
        self.octave = octave
        self.alter = alter

        has_pitch_spelling = not (
            self.step is None
            or self.octave is None
            or self.alter is None)

        if has_pitch_spelling:
            if self.midi_pitch != pitch_spelling_to_midi_pitch(
                    step=self.step,
                    octave=self.octave,
                    alter=self.alter):

                raise ValueError('The provided pitch spelling information does not match '
                                 'the given MIDI pitch')

    @property
    def duration(self):
        if self.use_sound_off:
            return self.sound_off - self.note_on
        else:
            return self.note_off - self.note_on

    def __str__(self):
        r = [self.__class__.__name__,
             '{0}: {1}'.format('MIDI pitch', self.midi_pitch),
             '{0}: {1}'.format('Onset time', self.note_on),
             '{0}: {1}'.format('Offset time', self.note_off),
             '{0}: {1}'.format('Sound off time', self.sound_off),
             '{0}: {1}'.format('Velocity', self.velocity)]

        return '\n'.join(r) + '\n'


class SustainPedal(object):
    def __init__(self, time, value):
        self.time = time
        self.value = value


class PerformedPart(object):
    """Represents a performed part, e.g.. all notes and related controller/modifiers of one single instrument

    Parameters
    ----------
    id : str
        The identifier of the part
    part_name : 
    """

    def __init__(self, id, part_name=None, notes=None,
                 pedal=None,
                 pedal_threshold=64,
                 midi_clock_units=4000,
                 midi_clock_rate=500000):
        super().__init__()
        self.id = id
        self.parent = None
        self.part_name = part_name

        if notes is None:
            notes = []
        if pedal is None:
            pedal = []
        self.notes = np.array(notes, dtype=PerformedNote)
        self.pedal = np.array(pedal, dtype=SustainPedal)

        self.pedal_threshold = pedal_threshold

        self.midi_clock_rate = float(midi_clock_rate)
        self.midi_clock_units = float(midi_clock_units)

    @property
    def _note_onsets_in_seconds(self):
        return np.array([float(n.note_on) * self.midi_clock_rate / (self.midi_clock_units * 1e6) for n in self.notes])

    @property
    def pedal_threshold(self):
        return self._pedal_threshold

    @pedal_threshold.setter
    def pedal_threshold(self, value):
        """
        Set the pedal threshold and update the sound_off
        of the notes
        """
        self._pedal_threshold = value

        adjust_offsets_w_sustain(notes=self.notes,
                                 sustain_pedals=self.pedal,
                                 threshold=self._pedal_threshold)

    # @property
    # def note_array(self):


def adjust_offsets_w_sustain(notes, sustain_pedals, threshold=64):
    # get all note offsets
    offs = np.array([n.note_off for n in notes])
    first_off = np.min(offs)
    last_off = np.max(offs)

    if len(sustain_pedals) > 0:
        # Get pedal times
        pedal = np.array([(x.time, x.value > threshold) for x in sustain_pedals])
        # sort, just in case
        pedal = pedal[np.argsort(pedal[:, 0]), :]

        # reduce the pedal info to just the times where there is a change in pedal state
        pedal = np.vstack(((min(pedal[0, 0] - 1, first_off - 1), 0),  # if there is an onset before the first pedal info, assume pedal is off
                           pedal[0, :],
                           pedal[np.where(np.diff(pedal[:, 1]) != 0)[0] + 1, :],
                           (max(pedal[-1, 0] + 1, last_off + 1), 0)  # if there is an offset after the last pedal info, assume pedal is off
                           ))
        last_pedal_change_before_off = np.searchsorted(pedal[:, 0], offs) - 1

        pedal_state_at_off = pedal[last_pedal_change_before_off, 1]
        pedal_down_at_off = pedal_state_at_off == 1
        next_pedal_time = pedal[last_pedal_change_before_off + 1, 0]

        offs[pedal_down_at_off] = next_pedal_time[pedal_down_at_off]

        for offset, note in zip(offs, notes):
            note.sound_off = offset
