
from madmom.io.midi import MIDIFile

import partitura.score as score

def load_midi(fn):
    mid = MIDIFile(fn, unit='ticks', timing='absolute')
    # print(mid.key_signatures[0])
    # print(mid.key_signatures.dtype)
    # return
    part_id = 'Part 1'
    sp = score.ScorePart(part_id)
    divs = score.Divisions(mid.ticks_per_beat)
    sp.timeline.add_starting_object(0, divs)

    # add notes
    for i, (onset, pitch, duration, velocity, channel) in enumerate(mid.notes):
        step, alter, octave = decode_pitch(pitch)
        note_id = f's{i:04d}'
        note = score.Note(step, alter, octave, id=note_id)
        sp.timeline.add_starting_object(int(onset), note)
        sp.timeline.add_ending_object(int(onset+duration), note)
        
    # time signatures and measures
    time_sigs = mid.time_signatures.astype(np.int)
    # for convenience we add the end times for each time signature
    ts_end_times = np.r_[time_sigs[1:, 0], np.iinfo(np.int).max]
    time_sigs = np.column_stack((time_sigs, ts_end_times))
    
    measure_counter = 0
    for ts_start, num, den, ts_end in time_sigs:

        time_sig = score.TimeSignature(num, den)

        sp.timeline.add_starting_object(ts_start, time_sig)

        measure_duration = (num * mid.ticks_per_beat * 4) // den
        measure_start_limit = min(ts_end, sp.timeline.last_point.t)

        for m_start in range(ts_start, measure_start_limit, measure_duration):
            measure = score.Measure(number=measure_counter)
            m_end = min(m_start+measure_duration, ts_end)
            sp.timeline.add_starting_object(m_start, measure)
            sp.timeline.add_ending_object(m_end, measure)
            measure_counter += 1

        if np.isinf(ts_end):
            ts_end = m_end
            
        sp.timeline.add_ending_object(max(ts_start, min(ts_end, m_end)), time_sig)

    sp.timeline.add_ending_object(sp.timeline.last_point.t, divs)
    return sp
