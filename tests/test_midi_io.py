


import ipdb
import os
import partitura as pa


output_fn = '/home/thassilo/gitrep/partitura/tests/data_examples/test_midi_output.mid'


# print(os.cwd())
# sp = midi.load_midi('./tests/data_examples/test_basic_midi.mid')

# load up musicxml file
# part_list = pa.load_musicxml('/home/thassilo/gitrep/partitura/tests/data_examples/test_basic_midi_pickup.musicxml')
# part_list = pa.load_musicxml('/home/thassilo/gitrep/partitura/tests/data_examples/test_basic_midi_2_manipulated_divs.musicxml')
# part_list = pa.load_musicxml('/home/thassilo/gitrep/partitura/tests/data_examples/test_basic_midi_2.musicxml')
part_list = pa.load_musicxml('/home/thassilo/gitrep/partitura/tests/data_examples/test_basic_midi_3.musicxml')


p1 = part_list[0]  # get score part from list

pa.save_midi(output_fn, part_list)




