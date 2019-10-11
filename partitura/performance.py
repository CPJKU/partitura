import numpy as np

class PerformedPart(object):
    """Represents a performed part, e.g.. all notes and related controller/modifiers of one single instrument

    Parameters
    ----------
    id : str
        The identifier of the part
    part_name : 
    """

    def __init__(self, notes, id=None, part_name=None, pedal=None, pedal_threshold=64):
        super().__init__()
        self.id = id
        self.part_name = part_name

        self.notes = np.array(notes)
        self.pedal = np.array(pedal or [])

        self.pedal_threshold = pedal_threshold

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
        adjust_offsets_w_sustain(self.notes, self.pedal, self._pedal_threshold)


def adjust_offsets_w_sustain(notes, sustain_pedals, threshold=64):
    # get all note offsets
    offs = np.fromiter((n['note_off'] for n in notes), dtype=np.float)
    first_off = np.min(offs)
    last_off = np.max(offs)

    if len(sustain_pedals) > 0:
        # Get pedal times
        print(sustain_pedals)
        pedal = np.array([(x['time'], x['value'] > threshold) for x in sustain_pedals])
        # sort, just in case
        pedal = pedal[np.argsort(pedal[:, 0]), :]

        # reduce the pedal info to just the times where there is a change in pedal state
        
        pedal = np.vstack(((min(pedal[0, 0] - 1, first_off - 1), 0),  
                           pedal[0, :],
                           # if there is an onset before the first pedal info, assume pedal is off
                           pedal[np.where(np.diff(pedal[:, 1]) != 0)[0] + 1, :],
                           # if there is an offset after the last pedal info, assume pedal is off
                           (max(pedal[-1, 0] + 1, last_off + 1), 0)  
                           ))
        last_pedal_change_before_off = np.searchsorted(pedal[:, 0], offs) - 1

        pedal_state_at_off = pedal[last_pedal_change_before_off, 1]
        pedal_down_at_off = pedal_state_at_off == 1
        next_pedal_time = pedal[last_pedal_change_before_off + 1, 0]

        offs[pedal_down_at_off] = next_pedal_time[pedal_down_at_off]

        for offset, note in zip(offs, notes):
            note['sound_off'] = offset
