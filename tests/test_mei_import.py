import numpy as np
from lxml import etree

import partitura.score as score

from partitura.io.importmei import (get_value_from_attribute,
                                    get_value_from_tag,
                                    validate_mei,
                                    el_tag,
                                    _parse_mdiv,
                                    _parse_score,
                                    _parse_section,
                                    _parse_staffGrp,
                                    xns, mns)

if __name__ == '__main__':
    
    mei = './data/mei/Mozart_k265_v1.mei'
    validate = True
    parser = etree.XMLParser(resolve_entities=False,
                             huge_tree=False,
                             remove_comments=True,
                             remove_blank_text=True,
                             ns_clean=True) # For some reason, ns_clean does not remove the namespace from tags

    document = etree.parse(mei, parser)

    if validate:
        validate_mei(document, debug=True)
    

    root = document.getroot()

    ns = '{{{0}}}'.format(etree.QName(root.tag).namespace)

    music_el = root.find(ns + 'music')


    structure = []
    for matter_el in music_el:
        if el_tag(matter_el) == 'front':
            print('front matter')
        elif el_tag(matter_el) == 'back':
            print('back matter')
        if el_tag(matter_el) == 'body':
            # an MEI can contain a collection of works
            # check if there is a single work or multiple movements/pieces
            for body_el in matter_el:

                if el_tag(body_el) == 'mdiv':
                    # parse mdiv
                    mdiv = _parse_mdiv(body_el)
                    structure.append(mdiv)
