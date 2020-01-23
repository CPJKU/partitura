import numpy as np
from lxml import etree

import pkg_resources

import partitura.score as score

from partitura.io.importmusicxml import (get_value_from_attribute,
                                         get_value_from_tag)

_MEI_SCHEMA = pkg_resources.resource_filename('partitura', 'assets/mei.rng')
_MEI_VALIDATOR = None

def validate_mei(mei, debug=False):
    """
    Validate an MEI file against a RelaxNG.

    Parameters
    ----------
    mei: str
        Path to XML file
    debug: bool, optional
        If True, raise an exception when the xml is invalid, and print out the
        cause. Otherwise just return True when the XML is valid and False otherwise

    Returns
    -------
    bool or None
        None if debug=True, True or False otherwise, signalling validity

    """
    global _MEI_VALIDATOR
    if not _MEI_VALIDATOR:
        rng_doc = etree.parse(_MEI_SCHEMA)
        _MEI_VALIDATOR = etree.RelaxNG(rng_doc)

    is_valid = _MEI_VALIDATOR.validate(mei)
    if debug:
        return _MEI_VALIDATOR.assertValid(mei)
    else:
        return _MEI_VALIDATOR.validate(mei)

def el_tag(el):
    """
    Get tag (i.e., name of element) without namespace

    Parameters
    ----------
    el : lxml._Element
        Element to be inspected

    Returns
    -------
    tag : str
        Tag of the element without namespace
    """
    return etree.QName(el).localname

def _parse_mdiv(mdiv_el):
    """
    Parse music divisions elements <mdiv> </mdiv>
    """
    mdivs = []
    for el in mdiv_el:

        if el_tag(el) == 'mdiv':
            mdiv = _parse_mdiv(el)
            mdivs.append(mdiv)
        elif el_tag(el) == 'score':

            score = _parse_score(el)

            mdivs.append(score)

    if len(mdivs) == 1:
        return mdivs[0]
    else:
        return mdivs


def _parse_score(score_el):
    
    section_start_t = 0
    sections = []
    parts = []

    
    for el in score_el:
        if el_tag(el) == 'scoreDef':
            for def_el in el:
                if el_tag(def_el) == 'staffGrp':
                    import pdb
                    pdb.set_trace()
                    part_id = get_value_from_attribute(def_el, xns('id'), str)

                    part_name = get_value_from_tag(def_el, 'label', str)
                    part = score.Part(part_id)

                    parts.append(part)
                    
        if el_tag(el) == 'section':
            section, section_end_t = _parse_section(el, section_start_t)
            section_start_t = section_end_t

            sections.append(section)

    return sections
def _parse_section(section_el, section_start_t=0):

    m_number = 0
    measures = []
    for el in section_el:
        if el_tag(s_el) == 'measure':
            measure = score.Measure(m_number)
            m_number += 1

            measures.append(measure)

    return measures, section_start_t

def _parse_staffGrp(staffGrp_el):

    staff_groups = []
    part_id = get_value_from_attribute(staffGrp_el, xns('id'), str)
    
    for el in staffGrp_el:
        if el_tag(el) == 'staffGrp':
            staff_group = _parse_staffGrp(el)
            staff_groups.append(staff_group)

        # TODO: braces and barlines
        if el_tag(el) == 'label':
            
            pass
        


def xml_ns_name(name):
    return '{http://www.w3.org/XML/1998/namespace}' + name
# shortcut
xns = xml_ns_name

def mei_ns_name(name):
    return '{http://www.music-encoding.org/ns/mei}' + name
#shortcut
mns = mei_ns_name

            
            
    

