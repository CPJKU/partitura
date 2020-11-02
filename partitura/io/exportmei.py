import partitura
import partitura.score as score
from lxml import etree
from partitura.utils.generic import partition
from partitura.utils.music import estimate_symbolic_duration
from copy import copy


nameSpace = "http://www.music-encoding.org/ns/mei"

xmlIdString = "{http://www.w3.org/XML/1998/namespace}id"

def extendKey(dictOfLists, key, value):
    """extend or create a list at the given key in the given dictionary

    Parameters
    ----------
    dictOfLists:    dictionary
        where all values are lists
    key:            self explanatory
    value:          self explanatory

    """

    if key in dictOfLists.keys():
        if isinstance(value, list):
            dictOfLists[key].extend(value)
        else:
            dictOfLists[key].append(value)
    else:
        dictOfLists[key] = (value if isinstance(value, list) else [value])




def calc_dur_dots_splitNotes_firstTempDur(note, measure, numToNumbase_ratio=1):
    """
    Notes have to be represented as a string of elemental notes (there is no notation for arbitrary durations)
    This function calculates this string (the durations of the elemental notes and their dot counts),
    whether the note crosses the measure and the temporal duration of the first elemental note

    Parameters
    ----------
    note:               score.GenericNote
        The note whose representation as a string of elemental notes is calculated
    measure:            score.Measure
        The measure which contains note
    numToNumbase_ratio: float, optional
        scales the duration of note according to whether or not it belongs to a tuplet and which one


    Returns
    -------
    dur_dots:       list of int pairs
        this describes the string of elemental notes that represent the note notationally
        every pair in the list contains the duration and the dot count of an elemental note and
        the list is ordered by duration in decreasing order
    splitNotes:     list or None
        an empty list if note crosses measure
        None if it doesn't
    firstTempDur:   int or None
        duration of first elemental note in partitura time
    """

    if measure=="pad":
        return [], None, None

    if isinstance(note, score.GraceNote):
        mainNote = note.main_note
        #HACK: main note should actually be always not None for a proper GraceNote
        if mainNote!=None:
            dur_dots,_,_ = calc_dur_dots_splitNotes_firstTempDur(mainNote, measure)
            dur_dots = [(2*dur_dots[0][0], dur_dots[0][1])]
        else:
            dur_dots = [(8,0)]
            note.id+="_missing_main_note"
        return dur_dots, None, None

    note_duration = note.duration



    splitNotes = None

    if note.start.t+note.duration>measure.end.t:
        note_duration = measure.end.t - note.start.t
        splitNotes = []

    quarterDur = measure.start.quarter
    fraction = numToNumbase_ratio*note_duration/quarterDur

    intPart = int(fraction)
    fracPart = fraction - intPart

    # calc digits of fraction in base2
    untiedDurations = []
    powOf_2 = 1

    while intPart>0:
        bit = intPart%2
        untiedDurations.insert(0,bit*powOf_2)
        intPart=intPart//2
        powOf_2*=2


    powOf_2 = 1/2

    while fracPart > 0:
        fracPart*=2
        bit = int(fracPart)
        fracPart-=bit
        untiedDurations.append(bit*powOf_2)
        powOf_2/=2


    dur_dots = []

    curr_dur = 0
    curr_dots = 0

    def addDD(dur_dots,dur,dots):
        dur_dots.append((int(4/dur),dots))

    for untiedDur in untiedDurations:
        if curr_dur!=0:
            if untiedDur==0:
                addDD(dur_dots, curr_dur, curr_dots)
                curr_dots=0
                curr_dur=0
            else:
                curr_dots+=1
        else:
            curr_dur = untiedDur

    if curr_dur!=0:
        addDD(dur_dots, curr_dur, curr_dots)

    firstTempDur = int(untiedDurations[0]*quarterDur)

    return dur_dots,splitNotes, firstTempDur



def insertElem_check(t, inbetweenNotesElems):
    """Check if something like a clef etc appears before time t

    Parameters
    ----------
    t:                      int
        time from a Timepoint
    inbetweenNotesElems:    list of InbetweenNotesElements
        a list of objects describing things like clefs etc

    Returns
    -------
    True if something like a clef etc appears before time t
    """

    for ine in inbetweenNotesElems:
        if ine.elem!=None and ine.elem.start.t<=t:
            return True

    return False



def partition_handleNone(func, iter, partitionAttrib):
    p = partition(func,iter)
    newKey = None


    if None in p.keys():
        raise KeyError("PARTITION ERROR: some elements of set do not have partition attribute \""+partitionAttrib+"\"")

    return p

def addChild(parent,childName):
    return etree.SubElement(parent,childName)

def setAttributes(elem, *list_attrib_val):
    for attrib_val in list_attrib_val:
        elem.set(attrib_val[0],str(attrib_val[1]))


def attribsOf_keySig(ks):
    """
    Returns values of a score.KeySignature object necessary for a MEI document

    Parameters
    ----------
    ks: score.KeySignature

    Returns
    -------
    fifths: string
        describes the circle of fifths
    mode:   string
        "major" or "minor"
    pname:  string
        pitch letter
    """

    key = ks.name
    pname = key[0].lower()
    mode = "major"

    if len(key)==2:
        mode="minor"

    fifths = str(abs(ks.fifths))

    if ks.fifths<0:
        fifths+="f"
    elif ks.fifths>0:
        fifths+="s"

    return fifths, mode, pname

def firstInstances_perPart(cls, parts, start=score.TimePoint(0), end=score.TimePoint(1)):
    """
    Returns the first instances of a class (multiple objects with same start time are possible) in each part

    Parameters
    ----------
    cls:    class
    parts:  list of score.Part
    start:  score.TimePoint, optional
        start of the range to search in
    end:    score.TimePoint, optional
        end of the range to search in

    Returns
    -------
    instances_perPart: list of list of instances of cls
        sublists might be empty
        if all sublists are empty, instances_perPart is empty
    """
    if not isinstance(start, list):
        start = [start]*len(parts)
    elif not len(parts)==len(start):
        raise ValueError("ERROR at firstInstances_perPart: start times are given as list with different size to parts list")

    if not isinstance(end, list):
        end = [end]*len(parts)
    elif not len(parts)==len(end):
        raise ValueError("ERROR at firstInstances_perPart: end times are given as list with different size to parts list")

    for i in range(len(parts)):
        if start[i]==None and end[i]!=None or start[i]!=None and end[i]==None:
            raise ValueError("ERROR at firstInstances_perPart: (start==None) != (end==None) (None elements in start have to be at same position as in end and vice versa)")

    instances_perPart=[]

    nonEmpty = False

    for i,p in enumerate(parts):
        s = start[i]
        e = end[i]

        if s==None:
            instances_perPart.append([])
            continue

        instances = list(p.iter_all(cls,s,e))

        if len(instances)==0:
            instances_perPart.append([])
            continue

        nonEmpty = True
        t = min(instances, key=lambda i:i.start.t).start.t
        instances_perPart.append([i for i in instances if t==i.start.t])

    if nonEmpty:
        return instances_perPart

    return []

def firstInstance_perPart(cls, parts, start=score.TimePoint(0), end=score.TimePoint(1)):
    """
    Reduce the result of firstInstances_perPart, a 2D list, to a 1D list
    If there are multiple first instances then program aborts with error message

    Parameters
    ----------
    cls:    class
    parts:  list of score.Part
    start:  score.TimePoint, optional
        start of the range to search in
    end:    score.TimePoint, optional
        end of the range to search in

    Returns
    -------
    fipp: list of instances of cls
        elements might be None
    """
    fispp = firstInstances_perPart(cls, parts, start, end)

    fipp = []

    for i,fis in enumerate(fispp):
        if len(fis)==0:
            fipp.append(None)
        elif len(fis)==1:
            fipp.append(fis[0])
        else:
            raise ValueError("Part "+parts[i].name,
            "ID "+parts[i].id,
            "has more than one instance of "+str(cls)+" at beginning t=0, but there should only be a single one")

    return fipp

def firstInstances(cls, part, start=score.TimePoint(0), end=score.TimePoint(1)):
    """
    Returns the first instances of a class (multiple objects with same start time are possible) in the part

    Parameters
    ----------
    cls:    class
    part:   score.Part
    start:  score.TimePoint, optional
        start of the range to search in
    end:    score.TimePoint, optional
        end of the range to search in

    Returns
    -------
    fis: list of instances of cls
        might be empty
    """
    fis = firstInstances_perPart(cls, [part], start, end)

    if len(fis)==0:
        return []

    return fis[0]

def firstInstance(cls, part, start=score.TimePoint(0), end=score.TimePoint(1)):
    """
    Reduce the result of firstInstance_perPart, a 1D list, to an element
    If there are multiple first instances then program aborts with error message

    Parameters
    ----------
    cls:    class
    part:   score.Part
    start:  score.TimePoint, optional
        start of the range to search in
    end:    score.TimePoint, optional
        end of the range to search in

    Returns
    -------
    fi: instance of cls or None
    """
    fi = firstInstance_perPart(cls, [part], start, end)

    if len(fi)==0:
        return None

    return fi[0]


def commonSignature(cls, sig_eql, parts, currentMeasures=None):
    """
    Calculate whether a list of parts has a common signature (as in key or time signature)

    Parameters
    ----------
    cls:                score.KeySignature or score.TimeSignature
    sig_eql:            function
        takes 2 signature objects as input and returns whether they are equivalent (in some sense)
    parts:              list of score.Part
    currentMeasures:    list of score.Measure, optional
        current as in the measures of the parts that are played at the same time and are processed

    Returns
    -------
    commonSig:  instance of cls
        might be None if there is no commonality between parts
    """
    sigs = None
    if currentMeasures!=None:
        #HACK:  measures should probably not contain "pad" at this point, but an actual dummy measure with start and end times?
        sigs = firstInstance_perPart(cls, parts, start=[cm.start if cm!='pad' else None for cm in currentMeasures], end=[cm.end if cm!='pad' else None for cm in currentMeasures])
    else:
        sigs = firstInstance_perPart(cls, parts)

    if sigs==None or len(sigs)==0 or None in sigs:
        return None

    commonSig = sigs.pop()

    for sig in sigs:
        if sig.start.t!=commonSig.start.t or not sig_eql(sig, commonSig):
            return None

    return commonSig

def verticalSlice(list_2d, index):
    """
    Returns elements of the sublists at index in a 1D list
    all sublists of list_2d have to have len > index
    """
    vslice = []

    for list_1d in list_2d:
        vslice.append(list_1d[index])

    return vslice

def timeSig_eql(ts1,ts2):
    """
    equivalence function for score.TimeSignature objects
    """
    return ts1.beats==ts2.beats and ts1.beat_type==ts2.beat_type

def keySig_eql(ks1,ks2):
    """
    equivalence function for score.KeySignature objects
    """
    return ks1.name==ks2.name and ks1.fifths==ks2.fifths

def idx(len_obj):
    return range(len(len_obj))

def attribsOf_Clef(clef):
    """
    Returns values of a score.Clef object necessary for a MEI document

    Parameters
    ----------
    clef: score.Clef

    Returns
    -------
    sign: string
        shape of clef (F,G, etc)
    line:
        which line to place clef on
    """
    sign = clef.sign

    if sign=="percussion":
        sign="perc"

    if clef.octave_change!=None and clef.octave_change!=0:
        place = "above"

        if clef.octave_change<0:
            place="below"

        return sign, clef.line, 1+7*abs(clef.octave_change), place

    return sign, clef.line

def create_staffDef(staffGrp, clef):
    """

    Parameters
    ----------
    staffGrp:   etree.SubElement
    clef:       score.Clef
    """
    staffDef = addChild(staffGrp,"staffDef")

    attribs = attribsOf_Clef(clef)
    setAttributes(staffDef,("n",clef.number),("lines",5),("clef.shape",attribs[0]),("clef.line",attribs[1]))
    if len(attribs)==4:
        setAttributes(staffDef,("clef.dis",attribs[2]),("clef.dis.place",attribs[3]))




def padMeasure(s, measure_perStaff, notes_withinMeasure_perStaff, autoRestCount):
    """
    Adds a fake measure ("pad") to the measures of the staff s and a score.Rest object to the notes

    Parameters
    ----------
    s:                              int
        staff number
    measure_perStaff:               dict of score.Measure objects
    notes_withinMeasure_perStaff:   dict of lists of score.GenericNote objects
    autoRestCount:                  int
        a counter for all the score.Rest objects that are created automatically

    Returns
    -------
    incremented auto rest counter
    """

    measure_perStaff[s]="pad"
    r = score.Rest(id="pR"+str(autoRestCount), voice=1)
    r.start = score.TimePoint(0)
    r.end = r.start

    extendKey(notes_withinMeasure_perStaff, s, r)
    return autoRestCount+1




class InbetweenNotesElement:
    """
    InbetweenNotesElements contain information on objects like clefs, keysignatures, etc
    within the score and how to process them

    Parameters
    ----------
    name:           string
        name of the element used in MEI
    attribNames:    list of strings
        names of the attributes of the MEI element
    attribValsOf:   function
        a function that returns the attribute values of elem
    container_dict: dict of lists of partitura objects
        the container containing the required elements is at staff
    staff:          int
        staff number
    skipIndex:      int
        init value for the cursor i (might skip 0)

    Attributes
    ----------
    name:           string
        name of the element used in MEI
    attribNames:    list of strings
        names of the attributes of the MEI element
    elem:           instance of partitura object
    attribValsOf:   function
        a function that returns the attribute values of elem
    container:      list of partitura objects
        the container where elem gets its values from
    i:              int
        cursor that keeps track of position in container
    """

    __slots__ = ["name","attribNames","attribValsOf","container","i","elem"]

    def __init__(self, name, attribNames, attribValsOf, container_dict, staff, skipIndex):
        self.name = name
        self.attribNames = attribNames
        self.attribValsOf = attribValsOf

        self.i=0
        self.elem=None

        if staff in container_dict.keys():
            self.container = container_dict[staff]
            if len(self.container)>skipIndex:
                self.elem = self.container[skipIndex]
                self.i=skipIndex
        else:
            self.container=[]

def chordRep(chords,chord_i):
        return chords[chord_i][0]

def handleBeam(openUp, parents):
    """
    Using a stack of MEI elements, opens and closes beams

    Parameters
    ----------
    openUp:     boolean
        flag that indicates whether to open or close recent beam
    parents:    list of etree.SubElement
        stack of MEI elements that contain the beam element

    Returns
    -------
    unchanged openUp value
    """
    if openUp:
        parents.append(addChild(parents[-1],"beam"))
    else:
        parents.pop()

    return openUp



def isChordInTuplet(chord_i, tupletIndices):
    """
    check if chord falls in the range of a tuplet

    Parameters
    ----------
    chord_i:        int
        index of chord within chords array
    tupletIndices:  list of int pairs
        contains the index ranges of all the tuplets in a measure of a staff

    Returns
    -------
    whether chord falls in the range of a tuplet
    """
    for start, stop in tupletIndices:
        if start<=chord_i and chord_i<=stop:
            return True

    return False

def calcNumToNumbaseRatio(chord_i, chords, tupletIndices):
    """
    calculates how to scale a notes duration with regard to the tuplet it is in

    Parameters
    ----------
    chord_i:        int
        index of chord within chords array
    chords:         list of list of score.GenericNote
        array of chords (which are lists of notes)
    tupletIndices:  list of int pairs
        contains the index ranges of all the tuplets in a measure of a staff

    Returns
    -------
    the num to numbase ratio of a tuplet (eg. 3 in 2 tuplet is 1.5)
    """
    rep = chords[chord_i][0]
    if not isinstance(rep,score.GraceNote) and isChordInTuplet(chord_i, tupletIndices):
        return rep.symbolic_duration["actual_notes"]/rep.symbolic_duration["normal_notes"]
    return 1

def processChord(chord_i, chords, inbetweenNotesElements, openBeam, autoBeaming, parents, dur_dots, splitNotes, firstTempDur, tupletIndices, ties, measure, layer, tuplet_idCounter, openTuplet, lastKeySig, noteAlterations, notes_nextMeasure_perStaff, next_dur_dots=None):
    """
    creates <note>, <chord>, <rest>, etc elements from chords
    also creates <beam>, <tuplet>, etc elements if necessary for chords objects
    also creates <clef>, <keySig>, etc elements before chord objects from inbetweenNotesElements

    Parameters
    ----------
    chord_i:                    int
        index of chord within chords array
    chords:                     list of list of score.GenericNote
        chord array
    inbetweenNotesElements:     list of InbetweenNotesElements
        check this to see if something like clef needs to get inserted before chord
    openBeam:                   boolean
        flag that indicates whether a beam is currently open
    autoBeaming:                boolean
        flag that determines if automatic beams should be created or if it is kept manual
    parents:                    list of etree.SubElement
        stack of MEI elements that contain the most recent beam element
    dur_dots:                   list of int pairs
        describes how the chord actually gets notated via tied notes, each pair contains the duration of the notated note and its dot count
    splitNotes:                 list
        this is either empty or None
        if None, nothing is done with this
        if an empty list, that means this chord crosses into the next measure and a chord is created for the next measure which is tied to this one
    firstTempDur:               int
        amount of ticks (as in partitura) of the first notated note
    tupletIndices:              list of int pairs
        the ranges of tuplets within the chords array
    ties:                       dict
        out parameter, contains pairs of IDs which need to be connected via ties
        this function also adds to that
    measure:                    score.Measure

    layer:                      etree.SubElement
        the parent element of the elements created here
    tuplet_idCounter:           int

    openTuplet:                 boolean
        describes if a tuplet is open or not
    lastKeySig:                 score.KeySignature
        the key signature this chord should be interpeted in
    noteAlterations:            dict
        contains the alterations of staff positions (notes) that are relevant for this chord
    notes_nextMeasure_perStaff: dict of lists of score.GenericNote
        out parameter, add the result of splitNotes into this
    next_dur_dots:              list of int pairs, optional
        needed for proper beaming

    Returns
    -------
    tuplet_idCounter:    int
        incremented if tuplet created
    openBeam:           boolean
        eventually modified if beam opened or closed
    openTuplet:         boolean
        eventually modified if tuplet opened or closed
    """

    chordNotes = chords[chord_i]
    rep = chordNotes[0]

    for ine in inbetweenNotesElements:
        if insertElem_check(rep.start.t, [ine]):
            # note should maybe be split according to keysig or clef etc insertion time, right now only beaming is disrupted
            if openBeam and autoBeaming:
                openBeam = handleBeam(False,parents)

            xmlElem = addChild(parents[-1], ine.name)
            attribVals = ine.attribValsOf(ine.elem)

            if ine.name=="keySig":
                lastKeySig = ine.elem

            if len(ine.attribNames)<len(attribVals):
                raise ValueError("ERROR at insertion of inbetweenNotesElements: there are more attribute values than there are attribute names for xml element "+ine.name)

            for nv in zip(ine.attribNames[:len(attribVals)], attribVals):
                setAttributes(xmlElem,nv)

            if ine.i+1>=len(ine.container):
                ine.elem = None
            else:
                ine.i+=1
                ine.elem = ine.container[ine.i]

    if isChordInTuplet(chord_i, tupletIndices):
        if not openTuplet:
            parents.append(addChild(parents[-1],"tuplet"))
            num = rep.symbolic_duration["actual_notes"]
            numbase = rep.symbolic_duration["normal_notes"]
            setAttributes(parents[-1], (xmlIdString,"t"+str(tuplet_idCounter)), ("num",num), ("numbase",numbase))
            tuplet_idCounter+=1
            openTuplet = True
    elif openTuplet:
        parents.pop()
        openTuplet = False

    def setDur_Dots(elem,dur_dots):
        dur,dots=dur_dots
        setAttributes(elem,("dur",dur))

        if dots>0:
            setAttributes(elem,("dots",dots))

    if isinstance(rep,score.Note):
        if autoBeaming:
            # for now all notes are beamed, however some rules should be obeyed there, see Note Beaming and Grouping

            # check to close beam
            if openBeam and (dur_dots[0][0]<8 or chord_i-1>=0 and type(rep)!=type(chordRep(chords, chord_i-1))):
                openBeam = handleBeam(False,parents)

            # check to open beam (maybe again)
            if not openBeam and dur_dots[0][0]>=8:
                # open beam if there are multiple "consecutive notes" which don't get interrupted by some element
                if len(dur_dots)>1 and not insertElem_check(rep.start.t+firstTempDur, inbetweenNotesElements):
                    openBeam=handleBeam(True,parents)

                # open beam if there is just a single note that is not the last one in measure and next note in measure is of same type and fits in beam as well, without getting interrupted by some element
                elif len(dur_dots)<=1 and chord_i+1<len(chords) and next_dur_dots[0][0]>=8 and type(rep)==type(chordRep(chords,chord_i+1)) and not insertElem_check(chordRep(chords,chord_i+1).start.t, inbetweenNotesElements):
                    openBeam = handleBeam(True,parents)
        elif openBeam and chord_i>0 and rep.beam!=chordRep(chords,chord_i-1).beam:
            openBeam = handleBeam(False,parents)

        if not autoBeaming and not openBeam and rep.beam!=None:
           openBeam = handleBeam(True,parents)

        def conditional_gracify(elem, rep, chord_i, chords):
            if isinstance(rep,score.GraceNote):
                grace = "unacc"

                if rep.grace_type == "appoggiatura":
                    grace = "acc"

                setAttributes(elem,("grace",grace))

                if rep.steal_proportion != None:
                    setAttributes(elem,("grace.time",str(rep.steal_proportion*100)+"%"))

                if chord_i==0 or not isinstance(chordRep(chords,chord_i-1), score.GraceNote):
                    chords[chord_i]=[copy(n) for n in chords[chord_i]]

                    for n in chords[chord_i]:
                        n.tie_next = n.main_note



        def createNote(parent, n, id, lastKeySig, noteAlterations):
            note=addChild(parent,"note")

            step = n.step.lower()
            setAttributes(note,(xmlIdString,id),("pname",step),("oct",n.octave))

            if n.articulations!=None and len(n.articulations)>0:
                artics = []

                translation={
                    "accent":           "acc",
                    "staccato":         "stacc",
                    "tenuto":           "ten",
                    "staccatissimo":    "stacciss",
                    'spiccato':         "spicc",
                     'scoop':           "scoop",
                     'plop':            "plop",
                     'doit':            "doit"
                }

                for a in n.articulations:
                    if a in translation.keys():
                        artics.append(translation[a])
                setAttributes(note,("artic"," ".join(artics)))

            sharps=['f','c','g','d','a','e','b']
            flats=list(reversed(sharps))

            staffPos = step+str(n.octave)

            alter = n.alter or 0

            def setAccid(note, acc, noteAlterations, staffPos, alter):
                if staffPos in noteAlterations.keys() and alter==noteAlterations[staffPos]:
                    return
                setAttributes(note, ("accid",acc))
                noteAlterations[staffPos]=alter

            # sharpen note if: is sharp, is not sharpened by key or prev alt
            # flatten note if: is flat, is not flattened by key or prev alt
            # neutralize note if: is neutral, is sharpened/flattened by key or prev alt

            # check if note is sharpened/flattened by prev alt or key
            if staffPos in noteAlterations.keys() and noteAlterations[staffPos]!=0 or lastKeySig.fifths>0 and step in sharps[:lastKeySig.fifths] or lastKeySig.fifths<0 and step in flats[:-lastKeySig.fifths]:
                if alter==0:
                    setAccid(note, "n", noteAlterations, staffPos, alter)
            elif alter>0:
                setAccid(note, "s", noteAlterations, staffPos, alter)
            elif alter<0:
                setAccid(note, "f", noteAlterations, staffPos, alter)


            return note



        if len(chordNotes)>1:
            chord = addChild(parents[-1],"chord")

            setDur_Dots(chord,dur_dots[0])

            conditional_gracify(chord, rep, chord_i, chords)

            for n in chordNotes:
                createNote(chord, n, n.id, lastKeySig, noteAlterations)


        else:
            note=createNote(parents[-1], rep, rep.id, lastKeySig, noteAlterations)
            setDur_Dots(note,dur_dots[0])

            conditional_gracify(note,rep, chord_i, chords)

        if len(dur_dots)>1:
            for n in chordNotes:
                ties[n.id]=[n.id]

            def create_splitUpNotes(chordNotes, i, parents, dur_dots, ties, rep):
                if len(chordNotes)>1:
                    chord = addChild(parents[-1],"chord")
                    setDur_Dots(chord,dur_dots[i])

                    for n in chordNotes:
                        id = n.id+"-"+str(i)

                        ties[n.id].append(id)
                        createNote(chord, n, id, lastKeySig, noteAlterations)
                else:
                    id = rep.id+"-"+str(i)

                    ties[rep.id].append(id)

                    note=createNote(parents[-1], rep, id, lastKeySig, noteAlterations)

                    setDur_Dots(note,dur_dots[i])

            for i in range(1,len(dur_dots)-1):
                if not openBeam and dur_dots[i][0]>=8:
                    openBeam = handleBeam(True,parents)

                create_splitUpNotes(chordNotes, i,parents,dur_dots,ties,rep)

            create_splitUpNotes(chordNotes, len(dur_dots)-1,parents,dur_dots,ties,rep)


        if splitNotes!=None:


            for n in chordNotes:
                splitNotes.append(score.Note(n.step,n.octave, id=n.id+"s"))



            if len(dur_dots)>1:
                for n in chordNotes:
                    ties[n.id].append(n.id+"s")
            else:
                for n in chordNotes:
                    ties[n.id]=[n.id, n.id+"s"]

        for n in chordNotes:
            if n.tie_next!=None:
                if n.id in ties.keys():
                    ties[n.id].append(n.tie_next.id)
                else:
                    ties[n.id]=[n.id, n.tie_next.id]

    elif isinstance(rep,score.Rest):
        if splitNotes!=None:
            splitNotes.append(score.Rest(id=rep.id+"s"))

        if measure=="pad" or measure.start.t == rep.start.t and measure.end.t == rep.end.t:
            rest = addChild(layer,"mRest")

            setAttributes(rest,(xmlIdString,rep.id))
        else:
            rest = addChild(layer,"rest")

            setAttributes(rest,(xmlIdString,rep.id))

            setDur_Dots(rest,dur_dots[0])

            for i in range(1,len(dur_dots)):
                rest=addChild(layer,"rest")

                id = rep.id+str(i)

                setAttributes(rest,(xmlIdString,id))
                setDur_Dots(rest,dur_dots[i])

    if splitNotes!=None:
        for sn in splitNotes:
            sn.voice = rep.voice
            sn.start = measure.end
            sn.end = score.TimePoint(rep.start.t+rep.duration)

            extendKey(notes_nextMeasure_perStaff, s, sn)

    return tuplet_idCounter, openBeam, openTuplet



def createScoreDef(measures, measure_i, parts, parent):
    """
    creates <scoreDef>

    Parameters
    ----------
    measures:   list of score.Measure
    measure_i:  int
        index of measure currently processed within measures
    parts:      list of score.Part
    parent:     etree.SubElement
        parent of <scoreDef>
    """
    referenceMeasures = verticalSlice(measures,measure_i)



    commonKeySig = commonSignature(score.KeySignature, keySig_eql, parts, referenceMeasures)
    commonTimeSig = commonSignature(score.TimeSignature, timeSig_eql,parts, referenceMeasures)

    scoreDef = None

    if commonKeySig!=None or commonTimeSig!=None:
        scoreDef = addChild(parent,"scoreDef")

    if commonKeySig!=None:
        fifths, mode, pname = attribsOf_keySig(commonKeySig)

        setAttributes(scoreDef,("key.sig",fifths),("key.mode", mode),("key.pname",pname))

    if commonTimeSig!=None:
        setAttributes(scoreDef,("meter.count",commonTimeSig.beats),("meter.unit",commonTimeSig.beat_type))

    return scoreDef


class MeasureContent:
    """
    Simply a bundle for all the data of a measure that needs to be processed for a MEI document

    Attributes
    ----------
    ties_perStaff:      dict of lists
    clefs_perStaff:     dict of lists
    keySigs_perStaff:   dict of lists
    timeSigs_perStaff:  dict of lists
    measure_perStaff:   dict of lists
    tuplets_perStaff:   dict of lists
    slurs:              list
    dirs:               list
    dynams:             list
    tempii:             list
    fermatas:           list
    """
    __slots__ = ["ties_perStaff","clefs_perStaff","keySigs_perStaff","timeSigs_perStaff","measure_perStaff","tuplets_perStaff","slurs","dirs","dynams","tempii","fermatas"]

    def __init__(self):
        self.ties_perStaff = {}
        self.clefs_perStaff = {}
        self.keySigs_perStaff = {}
        self.timeSigs_perStaff = {}
        self.measure_perStaff = {}
        self.tuplets_perStaff = {}

        self.slurs = []
        self.dirs=[]
        self.dynams=[]
        self.tempii=[]
        self.fermatas=[]


def extractFromMeasures(parts, measures, measure_i, staves_perPart, autoRestCount, notes_withinMeasure_perStaff):
    """
    Returns a bundle of data regarding the measure currently processed, things like notes, key signatures, etc
    Also creates padding measures, necessary for example, for staves of instruments which do not play in the current measure

    Parameters
    ----------
    parts:                          list of score.Part
    measures:                       list of score.Measure
    measure_i:                      int
        index of current measure within measures
    staves_perPart:                 dict of list of ints
        staff enumeration partitioned by part
    autoRestCount:                  int
        counter for the IDs of automatically generated rests
    notes_withinMeasure_perStaff:   dict of lists of score.GenericNote
        in and out parameter, might contain note objects that have crossed from previous measure into current one

    Returns
    -------
    autoRestCount:                  int
        incremented if score.Rest created
    currentMeasureContent:          MeasureContent
        bundle for all the data that is extracted from the currently processed measure
    """
    currentMeasureContent = MeasureContent()

    for part_i, part in enumerate(parts):
        m = measures[part_i][measure_i]

        if m=="pad":
            for s in staves_perPart[part_i]:
                autoRestCount = padMeasure(s, currentMeasureContent.measure_perStaff, notes_withinMeasure_perStaff, autoRestCount)

            continue



        def cls_withinMeasure(part, cls, measure, incl_subcls=False):
            return part.iter_all(cls, measure.start, measure.end, include_subclasses=incl_subcls)

        def cls_withinMeasure_list(part, cls, measure, incl_subcls=False):
            return list(cls_withinMeasure(part,cls,measure,incl_subcls))

        clefs_withinMeasure_perStaff_perPart = partition_handleNone(lambda c:c.number, cls_withinMeasure(part, score.Clef, m),"number")
        keySigs_withinMeasure = cls_withinMeasure_list(part,score.KeySignature, m)
        timeSigs_withinMeasure = cls_withinMeasure_list(part, score.TimeSignature, m)
        currentMeasureContent.slurs.extend(cls_withinMeasure(part, score.Slur, m))
        tuplets_withinMeasure = cls_withinMeasure_list(part, score.Tuplet, m)

        beat_map = part.beat_map

        def calc_tstamp(beat_map, t, measure):
            return beat_map(t)-beat_map(measure.start.t)+1

        for w in cls_withinMeasure(part, score.Words, m):
            tstamp=calc_tstamp(beat_map, w.start.t, m)
            currentMeasureContent.dirs.append((tstamp,w))

        for tempo in cls_withinMeasure(part, score.Tempo, m):
            tstamp=calc_tstamp(beat_map, tempo.start.t, m)
            currentMeasureContent.tempii.append((tstamp,staves_perPart[part_i][0],tempo))

        for fermata in cls_withinMeasure(part,score.Fermata,m):
            tstamp=calc_tstamp(beat_map,fermata.start.t,m)
            currentMeasureContent.fermatas.append((tstamp,fermata.ref.staff))

        for dynam in cls_withinMeasure(part, score.Direction, m, True):
            tstamp=calc_tstamp(beat_map, dynam.start.t, m)
            tstamp2=None

            if dynam.end!=None:
                measureCounter = measure_i
                while True:
                    if dynam.end.t<=measures[part_i][measureCounter].end.t:
                        tstamp2 = calc_tstamp(beat_map, dynam.end.t, measures[part_i][measureCounter])

                        tstamp2 = str(measureCounter-measure_i)+"m+"+str(tstamp2)

                        break
                    elif measureCounter+1>=len(measures[part_i]) or measures[part_i][measureCounter+1]=='pad':
                        raise ValueError("A score.Direction instance has an end time that exceeds actual non-padded measures")
                    else:
                        measureCounter+=1

            currentMeasureContent.dynams.append((tstamp,tstamp2,dynam))

        notes_withinMeasure_perStaff_perPart = partition_handleNone(lambda n:n.staff, cls_withinMeasure(part,score.GenericNote, m, True), "staff")

        for s in staves_perPart[part_i]:
            currentMeasureContent.keySigs_perStaff[s]=keySigs_withinMeasure
            currentMeasureContent.timeSigs_perStaff[s]=timeSigs_withinMeasure
            currentMeasureContent.tuplets_perStaff[s]=tuplets_withinMeasure

            if s not in notes_withinMeasure_perStaff_perPart.keys():
                autoRestCount = padMeasure(s, currentMeasureContent.measure_perStaff, notes_withinMeasure_perStaff, autoRestCount)

        for s,nwp in notes_withinMeasure_perStaff_perPart.items():
            extendKey(notes_withinMeasure_perStaff, s, nwp)
            currentMeasureContent.measure_perStaff[s]=m

        for s,cwp in clefs_withinMeasure_perStaff_perPart.items():
            currentMeasureContent.clefs_perStaff[s]=cwp


    return autoRestCount, currentMeasureContent

def createMeasure(section, measure_i, staves_sorted, notes_withinMeasure_perStaff, scoreDef, tuplet_idCounter, autoBeaming, lastKeySig_perStaff, currentMeasureContent):
    """
    creates a <measure> element within <section>
    also returns an updated id counter for tuplets and a dictionary of notes that cross into the next measure

    Parameters
    ----------
    section:                        etree.SubElement
    measure_i:                      int
        index of the measure created
    staves_sorted:                  list of ints
        a sorted list of the proper staff enumeration of the score
    notes_withinMeasure_perStaff:   dict of lists of score.GenericNote
        contains score.Note, score.Rest, etc objects of the current measure, partitioned by staff enumeration
        will be further partitioned and sorted by voice, time and type (score.GraceNote) and eventually gathered into
        a list of equivalence classes called chords
    scoreDef:                       etree.SubElement
    tuplet_idCounter:               int
        tuplets usually don't come with IDs, so an automatic counter takes care of that
    autoBeaming:                    boolean
        enables automatic beaming
    lastKeySig_perStaff:            dict of score.KeySignature
        keeps track of the keysignature each staff is currently in
    currentMeasureContent:          MeasureContent
        contains all sorts of data for the measure like tuplets, slurs, etc

    Returns
    -------
    tuplet_idCounter:               int
        incremented if tuplet created
    notes_nextMeasure_perStaff:     dict of lists of score.GenericNote
        score.GenericNote objects that cross into the next measure
    """
    measure=addChild(section,"measure")
    setAttributes(measure,("n",measure_i+1))

    ties_perStaff={}

    for s in staves_sorted:
        noteAlterations = {}

        staff=addChild(measure,"staff")

        setAttributes(staff,("n",s))

        notes_withinMeasure_perStaff_perVoice = partition_handleNone(lambda n:n.voice, notes_withinMeasure_perStaff[s], "voice")

        ties_perStaff_perVoice={}

        m = currentMeasureContent.measure_perStaff[s]

        tuplets=[]
        if s in currentMeasureContent.tuplets_perStaff.keys():
            tuplets = currentMeasureContent.tuplets_perStaff[s]

        lastKeySig=lastKeySig_perStaff[s]

        for voice,notes in notes_withinMeasure_perStaff_perVoice.items():
            layer=addChild(staff,"layer")

            setAttributes(layer,("n",voice))

            ties={}



            notes_partition=partition_handleNone(lambda n:n.start.t, notes, "start.t")

            chords = []



            for t in sorted(notes_partition.keys()):
                ns = notes_partition[t]

                if len(ns)>1:
                    type_partition = partition_handleNone(lambda n: isinstance(n,score.GraceNote),ns,"isGraceNote")

                    if True in type_partition.keys():
                        gns = type_partition[True]

                        gn_chords=[]

                        def scanBackwards(gns):
                            start = gns[0]

                            while isinstance(start.grace_prev, score.GraceNote):
                                start = start.grace_prev

                            return start

                        start = scanBackwards(gns)

                        def processGraceNote(n, gns):
                            if not n in gns:
                                raise ValueError("Error at forward scan of GraceNotes: a grace_next has either different staff, voice or starting time than GraceNote chain")
                            gns.remove(n)
                            return n.grace_next

                        while isinstance(start, score.GraceNote):
                            gn_chords.append([start])
                            start = processGraceNote(start, gns)

                        while len(gns)>0:
                            start = scanBackwards(gns)

                            i=0
                            while isinstance(start, score.GraceNote):
                                if i>=len(gn_chords):
                                    raise IndexError("ERROR at GraceNote-forward scanning: Difference in lengths of grace note sequences for different chord notes")
                                gn_chords[i].append(start)
                                start = processGraceNote(start, gns)
                                i+=1

                            if not i==len(gn_chords):
                                raise IndexError("ERROR at GraceNote-forward scanning: Difference in lengths of grace note sequences for different chord notes")

                        for gnc in gn_chords:
                            chords.append(gnc)

                    if not False in type_partition.keys():
                        raise KeyError("ERROR at ChordNotes-grouping: GraceNotes detected without additional regular Notes at same time; staff "+str(s))

                    regNotes =type_partition[False]



                    rep = regNotes[0]

                    for i in range(1,len(regNotes)):
                        n = regNotes[i]

                        if n.duration!=rep.duration:
                            raise ValueError("In staff "+str(s)+",",
                            "in measure "+str(m.number)+",",
                            "for voice "+str(voice)+",",
                            "2 notes start at time "+str(n.start.t)+",",
                            "but have different durations, namely "+n.id+" has duration "+str(n.duration)+" and "+rep.id+" has duration "+str(rep.duration),
                            "change to same duration for a chord or change voice of one of the notes for something else")
                        # HACK: unpitched notes are treated as Rests right now
                        elif not isinstance(rep,score.Rest) and not isinstance(n,score.Rest):
                            if rep.beam!=n.beam:
                                print("WARNING: notes within chords don't share the same beam",
                                "specifically note "+str(rep)+" has beam "+str(rep.beam),
                                "and note "+str(n)+" has beam "+str(n.beam),
                                "export still continues though")
                            elif set(rep.tuplet_starts)!=set(n.tuplet_starts) and set(rep.tuplet_stops)!=set(n.tuplet_stops):
                                print("WARNING: notes within chords don't share same tuplets, export still continues though")
                    chords.append(regNotes)
                else:
                    chords.append(ns)

            tupletIndices = []
            for tuplet in tuplets:
                ci = 0
                start = -1
                stop = -1
                while ci<len(chords):
                    for n in chords[ci]:
                        if tuplet in n.tuplet_starts:
                            start=ci
                            break
                    for n in chords[ci]:
                        if tuplet in n.tuplet_stops:
                            stop=ci
                            break

                    if start>=0 and stop>=0:
                        if not start<=stop:
                            raise ValueError("In measure "+str(measure_i+1)+",",
                            "in staff "+str(s)+",",
                            "["+str(tuplet)+"] stops before it starts?",
                            "start="+str(start+1)+"; stop="+str(stop+1))
                        tupletIndices.append((start,stop))
                        break

                    ci+=1



            parents = [layer]
            openBeam = False

            next_dur_dots, next_splitNotes, next_firstTempDur = calc_dur_dots_splitNotes_firstTempDur(chords[0][0],m, calcNumToNumbaseRatio(0, chords, tupletIndices))




            inbetweenNotesElements = [
                InbetweenNotesElement("clef", ["shape","line","dis","dis.place"], attribsOf_Clef, currentMeasureContent.clefs_perStaff, s, int(measure_i==0)),
                InbetweenNotesElement("keySig", ["sig","mode","pname","sig.showchange"], (lambda ks: attribsOf_keySig(ks)+("true",)), currentMeasureContent.keySigs_perStaff, s, int(scoreDef!=None)),
                InbetweenNotesElement("meterSig", ["count","unit"], lambda ts: (ts.beats, ts.beat_type), currentMeasureContent.timeSigs_perStaff, s, int(scoreDef!=None))
            ]

            openTuplet = False

            notes_nextMeasure_perStaff={}

            for chord_i in range(len(chords)-1):
                dur_dots,splitNotes, firstTempDur = next_dur_dots, next_splitNotes, next_firstTempDur
                next_dur_dots, next_splitNotes, next_firstTempDur = calc_dur_dots_splitNotes_firstTempDur(chordRep(chords,chord_i+1), m, calcNumToNumbaseRatio(chord_i+1, chords, tupletIndices))
                tuplet_idCounter, openBeam, openTuplet=processChord(chord_i, chords, inbetweenNotesElements, openBeam, autoBeaming, parents, dur_dots, splitNotes, firstTempDur, tupletIndices, ties, m, layer, tuplet_idCounter, openTuplet, lastKeySig, noteAlterations, notes_nextMeasure_perStaff, next_dur_dots)


            tuplet_idCounter,_,_=processChord(len(chords)-1, chords, inbetweenNotesElements, openBeam, autoBeaming, parents, next_dur_dots, next_splitNotes, next_firstTempDur, tupletIndices, ties, m, layer,tuplet_idCounter, openTuplet, lastKeySig, noteAlterations, notes_nextMeasure_perStaff)




            ties_perStaff_perVoice[voice]=ties

        ties_perStaff[s]=ties_perStaff_perVoice

    for fermata in currentMeasureContent.fermatas:
        tstamp=fermata[0]
        fermata_staff=fermata[1]

        f=addChild(measure,"fermata")
        setAttributes(f,("staff",fermata_staff),("tstamp",tstamp))

    for slur in currentMeasureContent.slurs:
        s = addChild(measure,"slur")
        if slur.start_note==None or slur.end_note==None:
            raise ValueError("Slur is missing start or end")
        setAttributes(s, ("staff",slur.start_note.staff), ("startid","#"+slur.start_note.id), ("endid","#"+slur.end_note.id))

    for tstamp,word in currentMeasureContent.dirs:
        d = addChild(measure, "dir")
        setAttributes(d,("staff",word.staff),("tstamp",tstamp))
        d.text = word.text

    #smufl individual notes start with E1
    #these are the last 2 digits of the codes
    metronomeCodes = {
        'breve': "D0",
        'whole': "D2",
        'half': "D3",
        'h': "D3",
        'quarter': "D5",
        'q': "D5",
        'eighth': "D7",
        'e': "D5",
        '16th': "D9",
        '32nd': "DB",
        '64th': "DD",
        '128th': "DF",
        '256th': "E1"
    }

    for tstamp, staff, tempo in currentMeasureContent.tempii:
        t = addChild(measure, "tempo")
        setAttributes(t, ("staff",staff),("tstamp",tstamp))

        unit = str(tempo.unit)

        dots = unit.count(".")

        unit = unit[:-dots]

        stringToBuild = [" <rend fontname=\"VerovioText\">&#xE1",metronomeCodes[unit or "q"],";"]

        for i in range(dots):
            stringToBuild.append("&#xE1E7;")

        stringToBuild.append("</rend> = ")
        stringToBuild.append(str(tempo.bpm))

        t.text = "".join(stringToBuild)

    for tstamp,tstamp2,dynam in currentMeasureContent.dynams:
        if isinstance(dynam, score.DynamicLoudnessDirection):
            d = addChild(measure, "hairpin")
            form = ("cres" if isinstance(dynam, score.IncreasingLoudnessDirection) else "dim")
            setAttributes(d, ("form",form))

            # duration can also matter for other dynamics, might want to move this out of branch
            if tstamp2!=None:
                setAttributes(d,("tstamp2",tstamp2))
        else:
            d = addChild(measure, "dynam")
            d.text = dynam.text

        setAttributes(d,("staff",dynam.staff),("tstamp",tstamp))









    for s,tps in ties_perStaff.items():

        for v,tpspv in tps.items():

            for ties in tpspv.values():

                for i in range(len(ties)-1):
                    tie = addChild(measure, "tie")

                    setAttributes(tie, ("staff",s), ("startid","#"+ties[i]), ("endid","#"+ties[i+1]))

    for s,k in currentMeasureContent.keySigs_perStaff.items():
        if len(k)>0:
            lastKeySig_perStaff[s]=max(k,key=lambda k:k.start.t)

    return tuplet_idCounter, notes_nextMeasure_perStaff

def unpackPartGroup(partGrp, parts=[]):
    """
    Recursively gather individual parts into a list, flattening the tree of parts so to say

    Parameters
    ----------
    partGrp:    score.PartGroup
    parts:      list of score.Part, optional

    Returns
    -------
    parts:      list of score.Part
    """
    for c in partGrp.children:
        if isinstance(c, score.PartGroup):
            unpackPartGroup(c, parts)
        else:
            parts.append(c)

    return parts





def save_mei(parts, autoBeaming=True, fileName = "testResult", titleText=None):
    """
    creates an MEI document based on the parts provided
    So far only <score> is used and not <part> which means all the parts are gathered in one whole score and
    no individual scores are defined for individual parts

    Parameters
    ----------
    parts:          score.Part, score.PartGroup or list of score.Part
    autoBeaming:    boolean, optional
        if all beaming has been done manually then set to False
        otherwise this flag can be used to enable automatic beaming (beaming rules are still in progess)
    fileName:       string, optional
        should not contain file extension, .mei will be added automatically
    titleText:      string, optional
        name of the piece, e.g. "Klaviersonate Nr. 14" or "WAP"
        if not provided, a title will be derived from fileName
    """

    if isinstance(parts, score.PartGroup):
        parts=unpackPartGroup(parts)
    elif isinstance(parts, score.Part):
        parts = [parts]


    mei = etree.Element("mei")

    meiHead=addChild(mei,"meiHead")
    music = addChild(mei,"music")



    meiHead.set("xmlns",nameSpace)
    fileDesc = addChild(meiHead,"fileDesc")
    titleStmt=addChild(fileDesc,"titleStmt")
    pubStmt=addChild(fileDesc,"pubStmt")
    title=addChild(titleStmt,"title")
    title.set("type","main")

    #derive a title for the piece from the fileName
    if titleText==None:
        cursor=len(fileName)-1
        while cursor>=0 and fileName[cursor]!="/":
            cursor-=1

        tmp = fileName[cursor+1:].split("_")
        tmp = [s[:1].upper()+s[1:] for s in tmp]
        title.text = " ".join(tmp)
    else:
        title.text=titleText

    body = addChild(music,"body")
    mdiv=addChild(body,"mdiv")
    mei_score=addChild(mdiv,"score")


    classesWithStaff = [score.GenericNote, score.Words, score.Direction]



    staves_perPart=[]

    stavesAreValid = True

    for p in parts:
        tmp = {staffedObj.staff for cls in classesWithStaff for staffedObj in p.iter_all(cls,include_subclasses=True)}
        tmp = tmp.union({clef.number for clef in p.iter_all(score.Clef)})
        staves_perPart.append(list(tmp))

        if None in staves_perPart[-1]:
            stavesAreValid=False
            staves_perPart[-1].remove(None)

            staves_perPart[-1].append((max(staves_perPart[-1]) if len(staves_perPart[-1])>0 else 0)+1)

        staves_perPart[-1].sort()

    if stavesAreValid:
        staves_sorted = sorted([s for staves in staves_perPart for s in staves])

        i=0

        while i+1<len(staves_sorted):
            if staves_sorted[i]==staves_sorted[i+1]:
                stavesAreValid=False
                break

            i+=1



    if not stavesAreValid:
        staves_perPart_backup = staves_perPart

        staves_sorted = []
        staves_perPart = []


        #ASSUMPTION: staves are >0
        maxStaff = 0
        for staves in staves_perPart_backup:
            if len(staves)==0:
                staves_perPart.append([])
            else:
                shift = [s+maxStaff for s in staves]

                maxStaff+=max(staves)

                staves_sorted.extend(shift)
                staves_perPart.append(shift)


        #staves_sorted.sort()

        maxStaff = 0
        for i,p in enumerate(parts):
            for cls in classesWithStaff:
                for staffObj in p.iter_all(cls,include_subclasses=True):
                    staffObj.staff = maxStaff + (staffObj.staff if staffObj.staff!=None else max(staves_perPart_backup[i]))

            for clef in p.iter_all(score.Clef):
                clef.number=maxStaff + (clef.number if clef.number!=None else max(staves_perPart_backup[i]))

            maxStaff+=(max(staves_perPart_backup[i]) if len(staves_perPart_backup[i])>0 else 0)

    measures = [list(parts[0].iter_all(score.Measure))]
    paddingRequired = False
    maxLength = len(measures[0])
    for i in range(1,len(parts)):
        m = list(parts[i].iter_all(score.Measure))

        if len(m) > maxLength:
            maxLength = len(m)

        if not paddingRequired:
            paddingRequired = (len(m)!=len(measures[0]))

        measures.append(m)








    scoreDef = createScoreDef(measures, 0, parts, mei_score)

    scoreDef_setup = scoreDef

    if scoreDef==None:
        scoreDef_setup = addChild(mei_score,"scoreDef")

    clefs_perPart=firstInstances_perPart(score.Clef, parts)



    for i in idx(clefs_perPart):
        clefs_perPart[i] = partition_handleNone(lambda c:c.number, clefs_perPart[i], "number")

    if len(clefs_perPart)==0:
        create_staffDef(staffGrp, score.Clef(sign="G",line=2, number=1, octave_change=0))
    else:
        for staves in staves_perPart:
            staffGrp = addChild(scoreDef_setup,"staffGrp")
            for s in staves:
                clefs = None

                for clefs_perStaff in clefs_perPart:
                    if s in clefs_perStaff.keys():
                        clefs = clefs_perStaff[s]
                        break

                if clefs!=None:
                    clef = clefs[0]
                    if len(clefs)!=1:
                        raise ValueError("ERROR at staffDef creation: Staff "+str(clef.number)+" starts with more than 1 clef at t=0")
                    create_staffDef(staffGrp, clef)
                else:
                    create_staffDef(staffGrp, score.Clef(sign="G",line=2, number=s, octave_change=0))


    section = addChild(mei_score,"section")



    measuresAreAligned = True
    if paddingRequired:
        cursors = [0]*len(measures)
        tempii = [None]*len(measures)

        while measuresAreAligned:
            compareMeasures = {}
            for i,m in enumerate(measures):
                if cursors[i]<len(m):
                    compareMeasures[i]=m[cursors[i]]
                    cursors[i]+=1


            if len(compareMeasures)==0:
                break

            compM_keys = list(compareMeasures.keys())

            new_tempii = firstInstance_perPart(score.Tempo, [p for i, p in enumerate(parts) if i in compM_keys], start=[cm.start for cm in compareMeasures.values()], end=[cm.end for cm in compareMeasures.values()])

            if len(new_tempii)==0:
                for k in compM_keys:
                    new_tempii.append(tempii[k])
            else:
                for i,nt in enumerate(new_tempii):
                    if nt==None:
                        new_tempii[i]=tempii[compM_keys[i]]
                    else:
                        tempii[compM_keys[i]]=nt

            def normDur(m):
                return (m.end.t-m.start.t)//m.start.quarter

            rep_i=0
            while rep_i<len(new_tempii) and new_tempii[rep_i]==None:
                rep_i+=1

            if rep_i==len(new_tempii):
                continue

            rep_dur = normDur(compareMeasures[compM_keys[rep_i]])*new_tempii[rep_i].bpm

            for i in range(rep_i+1,len(compM_keys)):
                nt = new_tempii[i]

                if nt==None:
                    continue

                m = compareMeasures[compM_keys[i]]
                dur = normDur(m)*new_tempii[i].bpm

                if dur!=rep_dur:
                    measuresAreAligned=False
                    break

    tuplet_idCounter = 0

    if measuresAreAligned:
        timeOffset = [0]*len(measures)

        if paddingRequired:
            for i, mp in enumerate(measures):
                ii=len(mp)
                timeOffset[i]=mp[ii-1].end.t
                while ii<maxLength:
                    mp.append("pad")
                    ii+=1

        notes_lastMeasure_perStaff = {}
        autoRestCount = 0

        notes_withinMeasure_perStaff = notes_lastMeasure_perStaff




        autoRestCount, currentMeasureContent = extractFromMeasures(parts, measures, 0, staves_perPart, autoRestCount, notes_withinMeasure_perStaff)

        lastKeySig_perStaff = {}

        for s,k in currentMeasureContent.keySigs_perStaff.items():
            lastKeySig_perStaff[s]= (min(k,key=lambda k:k.start.t) if len(k)>0 else None)

        tuplet_idCounter, notes_lastMeasure_perStaff =createMeasure(section, 0, staves_sorted, notes_withinMeasure_perStaff, scoreDef, tuplet_idCounter, autoBeaming, lastKeySig_perStaff, currentMeasureContent)



        for measure_i in range(1,len(measures[0])):
            notes_withinMeasure_perStaff = notes_lastMeasure_perStaff

            autoRestCount, currentMeasureContent=extractFromMeasures(parts, measures, measure_i, staves_perPart, autoRestCount, notes_withinMeasure_perStaff)

            scoreDef=createScoreDef(measures, measure_i, parts, section)

            tuplet_idCounter, notes_lastMeasure_perStaff = createMeasure(section, measure_i, staves_sorted, notes_withinMeasure_perStaff, scoreDef, tuplet_idCounter, autoBeaming, lastKeySig_perStaff, currentMeasureContent)









    (etree.ElementTree(mei)).write(fileName+".mei",pretty_print=True)

    # post processing step necessary
    # etree won't write <,> and & into an element's text
    with open(fileName+".mei") as result:
        text = list(result.read())
        newText=[]

        i=0
        while i<len(text):
            ch=text[i]
            if ch=="&":
                if text[i+1:i+4]==["l","t",";"]:
                    ch="<"
                    i+=4
                elif text[i+1:i+4]==["g","t",";"]:
                    ch=">"
                    i+=4
                elif text[i+1:i+5]==["a","m","p",";"]:
                    i+=5
                else:
                    i+=1
            else:
                i+=1

            newText.append(ch)

        newText="".join(newText)


    with open(fileName+".mei","w") as result:
        result.write(newText)



