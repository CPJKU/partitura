"""
This file contains test functions for MEI export
"""

import unittest

from tests import MEI_TESTFILES
from partitura import load_musicxml, load_mei
import partitura.score as score
from partitura.io.importmei import (
    _parse_mei,
    _ns_name,
    _handle_main_staff_group,
    _handle_layer_in_staff_in_measure,
    load_mei,
)
from lxml import etree
from xmlschema.names import XML_NAMESPACE


# class TestSaveMEI(unittest.TestCase):

#     def test_save_mei(self):

#         with open(EXAMPLE_MEI, 'r') as f:
#             target_mei = f.read()

#         mei = save_mei(load_musicxml(EXAMPLE_MUSICXML), title_text='score_example')
#         msg = "Export of MEI of file {} does not yield identical result".format(EXAMPLE_MEI)

#         self.assertTrue(mei.decode('utf-8') == target_mei, msg)


class TestImportMEI(unittest.TestCase):
    def test_main_part_group1(self):
        document, ns = _parse_mei(MEI_TESTFILES[5])
        main_partgroup_el = document.find(_ns_name("staffGrp", ns, True))
        part_list = _handle_main_staff_group(main_partgroup_el, ns)
        self.assertTrue(len(part_list) == 2)
        # first partgroup
        self.assertTrue(isinstance(part_list[0], score.PartGroup))
        self.assertTrue(part_list[0].group_symbol == "bracket")
        self.assertTrue(part_list[0].group_name is None)
        self.assertTrue(part_list[0].id == "sl1ipm2")
        # first partgroup first part
        self.assertTrue(part_list[0].children[0].id == "P1")
        self.assertTrue(part_list[0].children[0].part_name == "S")
        self.assertTrue(part_list[0].children[0]._quarter_durations[0] == 12)
        # first partgroup second part
        self.assertTrue(part_list[0].children[1].id == "P2")
        self.assertTrue(part_list[0].children[1].part_name == "A")
        self.assertTrue(part_list[0].children[1]._quarter_durations[0] == 12)
        # first partgroup third part
        self.assertTrue(part_list[0].children[2].id == "P3")
        self.assertTrue(part_list[0].children[2].part_name == "T")
        self.assertTrue(part_list[0].children[2]._quarter_durations[0] == 12)
        # first partgroup fourth part
        self.assertTrue(part_list[0].children[3].id == "P4")
        self.assertTrue(part_list[0].children[3].part_name == "B")
        self.assertTrue(part_list[0].children[3]._quarter_durations[0] == 12)
        # second partgroup
        self.assertTrue(isinstance(part_list[1], score.PartGroup))
        self.assertTrue(part_list[1].group_symbol == "brace")
        self.assertTrue(part_list[1].group_name == "Piano")
        self.assertTrue(part_list[1].id == "P5")

    def test_main_part_group2(self):
        document, ns = _parse_mei(MEI_TESTFILES[4])
        main_partgroup_el = document.find(_ns_name("staffGrp", ns, True))
        part_list = _handle_main_staff_group(main_partgroup_el, ns)
        self.assertTrue(len(part_list) == 1)
        self.assertTrue(isinstance(part_list[0], score.PartGroup))

    def test_handle_layer1(self):
        document, ns = _parse_mei(MEI_TESTFILES[5])
        layer_el = [
            e
            for e in document.findall(_ns_name("layer", ns, True))
            if e.attrib[_ns_name("id", XML_NAMESPACE)] == "l3ss4q5"
        ][0]
        part = score.Part("dummyid", quarter_duration=12)
        _handle_layer_in_staff_in_measure(layer_el, 1, 1, 0, part, ns)
        self.assertTrue(len(part.note_array) == 3)

    def test_handle_layer2(self):
        document, ns = _parse_mei(MEI_TESTFILES[5])
        layer_el = [
            e
            for e in document.findall(_ns_name("layer", ns, True))
            if e.attrib[_ns_name("id", XML_NAMESPACE)] == "l95j799"
        ][0]
        part = score.Part("dummyid", quarter_duration=12)
        _handle_layer_in_staff_in_measure(layer_el, 1, 1, 0, part, ns)
        self.assertTrue(len(part.note_array) == 3)

    def test_handle_layer_tuplets(self):
        document, ns = _parse_mei(MEI_TESTFILES[6])
        layer_el = [
            e
            for e in document.findall(_ns_name("layer", ns, True))
            if e.attrib[_ns_name("id", XML_NAMESPACE)] == "l7hooah"
        ][0]
        part = score.Part("dummyid", quarter_duration=15)
        _handle_layer_in_staff_in_measure(layer_el, 1, 1, 0, part, ns)
        self.assertTrue(len(part.note_array) == 10)

    def test_parse_mei(self):
        part_list = load_mei(MEI_TESTFILES[6])
        self.assertTrue(len(part_list[0].children[0].note_array) == 10)

    def test_ties1(self):
        part_list = load_mei(MEI_TESTFILES[7])
        note_array = list(score.iter_parts(part_list))[0].note_array
        self.assertTrue(len(note_array) == 4)

    def test_time_signatures(self):
        part_list = load_mei(MEI_TESTFILES[8])
        part0 = list(score.iter_parts(part_list))[0]
        time_signatures = list(part0.iter_all(score.TimeSignature))
        self.assertTrue(len(time_signatures) == 3)
        self.assertTrue(time_signatures[0].t.start == 0)
        self.assertTrue(time_signatures[1].t.start == 8 * 16)
        self.assertTrue(time_signatures[2].t.start == 12.5 * 16)


if __name__ == "__main__":
    unittest.main()

