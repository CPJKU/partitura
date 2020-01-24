import partitura
import partitura.score as score
from lxml import etree
from partitura.utils.generic import partition
from partitura.utils.music import estimate_symbolic_duration
import sys


def errorPrint(*list_errorMsg):
    for msg in list_errorMsg:
        print(msg)

    sys.exit()


# parts = [score.Part("P0","Test"), score.Part("P1","Test"), score.Part("P2","Test")]
#
#
# parts[0].set_quarter_duration(0,2)
# parts[0].add(score.KeySignature(0,"major"),start=0)
# parts[0].add(score.TimeSignature(4,4),start=0)
# parts[1].set_quarter_duration(0,2)
# parts[1].add(score.KeySignature(0,"major"),start=0)
# parts[1].add(score.TimeSignature(4,4),start=0)
# parts[2].set_quarter_duration(0,2)
# parts[2].add(score.KeySignature(0,"major"),start=0)
# parts[2].add(score.TimeSignature(4,4),start=0)
#
# parts[0].add(score.Clef(sign="G",line=2, octave_change=0, number=1),start=0)
# parts[1].add(score.Clef(sign="F",line=4, octave_change=0, number=2),start=0)
# parts[2].add(score.Clef(sign="G",line=2, octave_change=0, number=3),start=0)
#
#
#
# parts[0].add(score.Note(id="n0s2",step="C",octave=4, staff=1, voice=1),start=0,end=5)
# parts[0].add(score.Note(id="n2s2",step="G",octave=4, staff=1, voice=1),start=5,end=6)
# parts[0].add(score.Note(id="n3s2",step="E",octave=4, staff=1, voice=1),start=6,end=7)
# parts[0].add(score.Note(id="n4s2",step="F",octave=4, staff=1, voice=1),start=7,end=9)
# parts[0].add(score.Note(id="n5s2",step="D",octave=4, staff=1, voice=1),start=9,end=10)
# parts[0].add(score.Note(id="n6s2",step="A",octave=4, staff=1, voice=1),start=10,end=16)
#
# parts[2].add(score.Note(id="n0s22",step="E",octave=4, staff=3, voice=1),start=0,end=5)
# parts[2].add(score.Note(id="n2s22",step="B",octave=4, staff=3, voice=1),start=5,end=6)
# parts[2].add(score.Note(id="n3s22",step="G",octave=4, staff=3, voice=1),start=6,end=7)
# parts[2].add(score.Note(id="n4s22",step="A",octave=4, staff=3, voice=1),start=7,end=9)
# parts[2].add(score.Note(id="n5s22",step="F",octave=4, staff=3, voice=1),start=9,end=10)
# parts[2].add(score.Note(id="n6s22",step="C",octave=5, staff=3, voice=1),start=10,end=16)
#
# parts[1].add(score.Note(id="n0",step="C",octave=2, staff=2, voice=1),start=0,end=3)
# parts[1].add(score.Note(id="n2",step="E",octave=2, staff=2, voice=1),start=3,end=6)
# parts[1].add(score.Note(id="n3",step="G",octave=2, staff=2, voice=1),start=6,end=7)
# parts[1].add(score.Note(id="n4",step="D",octave=2, staff=2, voice=1),start=7,end=10)
# parts[1].add(score.Note(id="n5",step="F",octave=2, staff=2, voice=1),start=10,end=13)
# parts[1].add(score.Note(id="n6",step="A",octave=2, staff=2, voice=1),start=13,end=16)
# score.add_measures(parts[0])
# score.add_measures(parts[1])
# score.add_measures(parts[2])
#
# score.tie_notes(parts[0])
# score.tie_notes(parts[1])
# test case blues lick
# part = score.Part("P0","Test")
# part.set_quarter_duration(0,10)
# part.add(score.KeySignature(-3,"minor"),start=0)
# part.add(score.TimeSignature(6,8),start=0)
# part.add(score.Clef(sign="F",line=4, octave_change=0, number=1),start=0)
# part.add(score.Clef(sign="G",line=2, octave_change=0, number=2),start=0)
# n0 = score.Note(id="n0",step="C",octave=2,voice=1, staff=1)
# n1 =score.Note(id="n1",step="E",octave=2,voice=1, staff=1, alter=-1)
# n2 =score.Note(id="n2",step="D",octave=2,voice=1, staff=1)
# part.add(n0,start=0,end=5)
# part.add(n1,start=5,end=10)
# part.add(n2,start=10,end=15)
# n0q = score.Note(id="n0q",step="G",octave=2,voice=1, staff=1)
# n1q =score.Note(id="n1q",step="G",octave=2,voice=1, staff=1)
# n2q =score.Note(id="n2q",step="G",octave=2,voice=1, staff=1)
# part.add(n0q,start=0,end=5)
# part.add(n1q,start=5,end=10)
# part.add(n2q,start=10,end=15)
# part.add(score.Note(id="n3",step="C",octave=2,voice=1, staff=1),start=15,end=40)
# n0s2 = score.Note(id="n0s2",step="C",octave=4,voice=1, staff=2)
# n1s2 =score.Note(id="n1s2",step="E",octave=4,voice=1, staff=2, alter=-1)
# n2s2 =score.Note(id="n2s2",step="D",octave=4,voice=1, staff=2)
# part.add(n0s2,start=0,end=5)
# part.add(n1s2,start=5,end=10)
# part.add(n2s2,start=10,end=15)
# part.add(score.Slur(n0s2,n2s2),start=0)
# part.add(score.Note(id="n3s2",step="C",octave=4,voice=1, staff=2),start=15,end=40)
#
# score.add_measures(part)
# score.tie_notes(part)
# parts=[part]

# # testing crossing measures and tieing notes together
# part.set_quarter_duration(0,16)
# part.add(score.KeySignature(-3,"minor"),start=0)
# part.add(score.TimeSignature(4,4),start=0)
# part.add(score.Clef(sign="F",line=4, octave_change=0, number=1),start=0)
#
# part.add(score.Rest(id="r0",staff=1, voice=1),start=0,end=1)
# part.add(score.Note(id="n0",step="C",octave=2, staff=1, voice=1),start=1,end=1+3*16+4+1)
# part.add(score.Note(id="n2",step="G",octave=2, staff=1, voice=1),start=1,end=1+3*16+4+1)
# part.add(score.Note(id="n3",step="E",octave=3, staff=1, voice=1),start=1,end=1+3*16+4+1)
#
#

# using this feature?
# making ties then becomes about looking at the tiegroup tag of notes
# which is fine, however the sum of powers of 2 idea seems better than estimating symbolic duration
# without this feature, notes crossing measure boundaries have to be handled
#score.tie_notes(part)



parts = partitura.load_musicxml("../../tests/data/musicxml/test_note_ties.xml", force_note_ids=True)
partitura.render(parts)

# part = parts
#
# qd=part.quarter_durations()[0][1]
#
# part.add(score.Clef(sign="F",line=4, octave_change=0, number=2), start=int(qd*(2+1/4)))
# part.add(score.KeySignature(-3,"minor"),start=int(qd*(2+1/4)))


if isinstance(parts, score.PartGroup):
    parts = parts.children
else:
    parts=[parts]



def addChild(parent,childName):
    return etree.SubElement(parent,childName)

def setAttributes(elem, *list_attrib_val):
    for attrib_val in list_attrib_val:
        elem.set(attrib_val[0],str(attrib_val[1]))

nameSpace = "http://www.music-encoding.org/ns/mei"

xmlIdString = "{http://www.w3.org/XML/1998/namespace}id"



mei = etree.Element("mei")

meiHead=addChild(mei,"meiHead")
music = addChild(mei,"music")



meiHead.set("xmlns",nameSpace)
fileDesc = addChild(meiHead,"fileDesc")
titleStmt=addChild(fileDesc,"titleStmt")
pubStmt=addChild(fileDesc,"pubStmt")
title=addChild(titleStmt,"title")
title.set("type","main")
title.text="TEST"

body = addChild(music,"body")
mdiv=addChild(body,"mdiv")
mei_score=addChild(mdiv,"score")

scoreDef = addChild(mei_score,"scoreDef")






def attribsOf_keySig(ks):
    key = ks.name
    pname = key[0]
    mode = "major"

    if len(key)==2:
        mode="minor"

    fifths = str(abs(ks.fifths))

    if ks.fifths<0:
        fifths+="f"
    else:
        fifths+="s"

    return fifths, mode, pname

def equal_keySig(ks1, ks2):
    return ks1.name==ks2.name and ks1.fifths==ks2.fifths

def firstInstance(cls, part, singleInstance=True):
    instances = list(part.iter_all(cls, score.TimePoint(0), score.TimePoint(1)))

    if singleInstance:
        if len(instances)>1:
            errorPrint("Part "+part.name,
            "ID "+part.id,
            "has more than one instance of "+str(cls)+" at beginning t=0, but there should only be a single one")
        else:
            if len(instances)>0:
                return instances[0]
            else:
                return None
    else:
        return instances

commonKeySig = firstInstance(score.KeySignature, parts[0])
commonTimeSig = firstInstance(score.TimeSignature, parts[0])

for i in range(1,len(parts)):
    p = parts[i]

    ks = firstInstance(score.KeySignature, p)
    ts = firstInstance(score.TimeSignature, p)

    if not (commonKeySig==None or ks.name==commonKeySig.name and ks.fifths==commonKeySig.fifths):
        commonKeySig=None

    if not (commonTimeSig==None or ts.beats==commonTimeSig.beats and ts.beat_type==commonTimeSig.beat_type):
        commonTimeSig=None



if commonKeySig!=None:
    fifths, mode, pname = attribsOf_keySig(commonKeySig)

    setAttributes(scoreDef,("key.sig",fifths),("key.mode", mode),("key.pname",pname))

if commonTimeSig!=None:
    setAttributes(scoreDef,("meter.count",commonTimeSig.beats),("meter.unit",commonTimeSig.beat_type))

section = addChild(mei_score,"section")

# might want to count staff numbers during processing and update staffGrp if count isn't consistent with clefs
staffGrp = addChild(scoreDef,"staffGrp")

def partition_handleNone(func, iter, partitionAttrib):
    p = partition(func,iter)
    newKey = None

    if None in p.keys():
        #errorPrint("PARTITION ERROR: some elements of set do not have partition attribute \""+partitionAttrib+"\"")

        # for testing purposes, introduce phantom staff, however, return to error when done testing
        newKey = 1

        for k in p.keys():
            if k!=None and k>newKey:
                newKey=k
        p[newKey]=p[None]
        del p[None]

    return p

clefs=[]

for p in parts:
    clefs.extend(firstInstance(score.Clef, p, singleInstance=False))

clefs = partition_handleNone(lambda c:c.number, clefs, "number")

if len(clefs)==0:
    staffDef = addChild(staffGrp,"staffDef")
    setAttributes(staffDef,("n",1),("lines",5),("clef.shape","G"),("clef.line",2))
else:
    for c in clefs.values():
        if len(c)>1:
            errorPrint("ERROR: Staff "+n+" starts with more than 1 clef at t=0")

        clef = c[0]

        staffDef = addChild(staffGrp,"staffDef")
        setAttributes(staffDef,("n",clef.number),("lines",5),("clef.shape",clef.sign),("clef.line",clef.line))



# check if all measures align

measures = [list(parts[0].iter_all(score.Measure))]
for i in range(1,len(parts)):
    m = list(p.iter_all(score.Measure))

    # if len(m)!=len(measures[0]), does that mean MEI file can't be structured with measures or that staffs with fewer measures have to be padded

    measures.append(m)

def isTemporallyAligned(measures):
    i=0

    while True:
        if i<len(measures[0]):
            rep = measures[0][i]
            duration = (rep.end.t-rep.start.t)//rep.start.quarter
            for ii in range(1,len(parts)):
                # bounds check might change depending if all parts have to have same amount of measures
                if i<len(measures[ii]):
                    m = measures[ii][i]

                    dur = (m.end.t-m.start.t)//m.start.quarter

                    if dur!=duration:
                        return False

            i+=1

        else:
            # break might change depending if all parts have to have same amount of measures
            break

    return True


notes_lastMeasure_perStaff = {}


for measure_i in range(len(measures[0])):
    measure=addChild(section,"measure")
    setAttributes(measure,("n",measure_i))

    notes_nextMeasure_perStaff={}
    ties_perStaff = {}

    for part_i in range(len(parts)):
        m = measures[part_i][measure_i]
        clefs_withinMeasure_perStaff = partition_handleNone(lambda c:c.number, parts[part_i].iter_all(score.Clef, m.start, m.end),"number")
        keySigs_withinMeasure = list(parts[part_i].iter_all(score.KeySignature, m.start, m.end))
        timeSigs_withinMeasure = list(parts[part_i].iter_all(score.TimeSignature, m.start, m.end))


        quarterDur = m.start.quarter

        notes_withinMeasure_perStaff = partition_handleNone(lambda n:n.staff, parts[part_i].iter_all(score.GenericNote, m.start, m.end, include_subclasses=True), "staff")

        notes_lastMeasure_perStaff_keys=set(notes_lastMeasure_perStaff.keys())
        notes_withinMeasure_perStaff_keys=set(notes_withinMeasure_perStaff.keys())

        staff_intersection_keys = notes_lastMeasure_perStaff_keys.intersection(notes_withinMeasure_perStaff_keys)

        for s in staff_intersection_keys:
            notes_withinMeasure_perStaff[s].extend(notes_lastMeasure_perStaff[s])

        def create_staff(notes_perStaff, ties_perStaff, notes_nextMeasure_perStaff):
            for s in notes_perStaff.keys():
                staff=addChild(measure,"staff")

                setAttributes(staff,("n",s))

                notes_withinMeasure_perStaff_perVoice = partition_handleNone(lambda n:n.voice, notes_perStaff[s], "voice")

                ties_perStaff_perVoice={}

                clefs_withinMeasure = None
                if s in clefs_withinMeasure_perStaff.keys():
                    clefs_withinMeasure = clefs_withinMeasure_perStaff[s]

                for voice,notes in notes_withinMeasure_perStaff_perVoice.items():
                    layer=addChild(staff,"layer")

                    setAttributes(layer,("n",voice))

                    ties={}

                    notes.sort(key=lambda n:n.start.t)

                    chords = [[notes[0]]]



                    for i in range(1,len(notes)):
                        rep = chords[-1][0]
                        n = notes[i]
                        # should multiple Rests with same start time or a mix of Rests and Notes with same start times be ignored, trigger a warning or be an error?
                        if isinstance(rep,score.Note) and isinstance(n, score.Note) and rep.start.t==n.start.t:
                            if rep.duration==n.duration:
                                chords[-1].append(n)
                            else:
                                errorPrint("In staff "+str(s)+",",
                                "in measure "+str(m.number)+",",
                                "for voice "+str(voice)+",",
                                "2 notes start at time "+str(n.start.t)+",",
                                "but have different durations, namely "+n.id+" has duration "+str(n.duration)+" and "+rep.id+" has duration "+str(rep.duration),
                                "change to same duration for a chord or change voice of one of the notes for something else")

                        else:
                            chords.append([n])

                    def calc_dur_dots_splitNotes_firstTempDur(n):


                        note_duration = n.duration

                        splitNotes = None

                        if n.start.t+n.duration>m.end.t:
                            note_duration = m.end.t - n.start.t
                            splitNotes = []


                        # what if note doesn't have any ID?
                        # maybe n.id = generateId()

                        fraction = note_duration/quarterDur
                        intPart = int(fraction)
                        fracPart = fraction - intPart



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




                        def powerOf2_toDur(p):
                            return int(4/p)

                        dur_dots = []

                        curr_dur = 0
                        curr_dots = 0

                        i=0

                        while i<len(untiedDurations):
                            if curr_dur!=0:
                                if untiedDurations[i]==0:
                                    dur_dots.append((powerOf2_toDur(curr_dur), curr_dots))
                                    curr_dots=0
                                    curr_dur=0
                                else:
                                    curr_dots+=1
                            else:
                                curr_dur = untiedDurations[i]

                            i+=1

                        if curr_dur!=0:
                            dur_dots.append((powerOf2_toDur(curr_dur), curr_dots))

                        firstTempDur = int(untiedDurations[0]*quarterDur)

                        return dur_dots,splitNotes, firstTempDur



                    openBeam=False
                    parent = layer

                    next_dur_dots, next_splitNotes, next_firstTempDur = calc_dur_dots_splitNotes_firstTempDur(chords[0][0])

                    def firstValidItem(container, notEmpty):
                        if notEmpty:
                            if measure_i==0:
                                if len(container)>1:
                                    return container[1]
                            else:
                                return container[0]

                        return None

                    clef_i=0
                    clef = firstValidItem(clefs_withinMeasure, s in clefs_withinMeasure_perStaff.keys())

                    keySig_i=0
                    keySig = firstValidItem(keySigs_withinMeasure, len(keySigs_withinMeasure)>0)

                    timeSig_i=0
                    timeSig = firstValidItem(timeSigs_withinMeasure, len(timeSigs_withinMeasure)>0)

                    def insertElem_check(t, *list_elem):
                        for elem in list_elem:
                            if elem!=None and elem.start.t<=t:
                                return True

                        return False

                    def insertElem(elem, note, elemName, attribNames, attribValsOf_elem, cursor, elemContainer):
                        ob = openBeam
                        p = parent
                        if insertElem_check(note.start.t, elem):
                            # note should maybe be split according to keysig or clef etc insertion time, right now only beaming is disrupted
                            if openBeam:
                                ob = False
                                p = layer

                            xmlElem = addChild(p, elemName)
                            attribVals = attribValsOf_elem(elem)

                            for nv in zip(attribNames, attribVals):
                                setAttributes(xmlElem,nv)

                            cursor+=1

                            if cursor==len(elemContainer):
                                elem = None
                            else:
                                elem = elemContainer[cursor]

                        return ob, p, cursor, elem


                    for chord_i in range(len(chords)):
                        chordNotes = chords[chord_i]
                        rep = chordNotes[0]
                        dur_dots,splitNotes, firstTempDur = next_dur_dots, next_splitNotes, next_firstTempDur

                        openBeam, parent, clef_i, clef = insertElem(clef, rep, "clef", ["shape","line"], lambda c:(c.sign,c.line), clef_i, clefs_withinMeasure)

                        openBeam, parent, keySig_i, keySig = insertElem(keySig, rep, "keySig", ["sig","mode","pname"], attribsOf_keySig, keySig_i, keySigs_withinMeasure)

                        openBeam, parent, timeSig_i, timeSig = insertElem(timeSig, rep, "meterSig", ["count","unit"], lambda ts: (ts.beats, ts.beat_type), timeSig_i, timeSigs_withinMeasure)



                        # hack right now, don't need to check every iteration, good time to factor out inside of loop
                        if chord_i < len(chords)-1:
                            next_dur_dots, next_splitNotes, next_firstTempDur = calc_dur_dots_splitNotes_firstTempDur(chords[chord_i+1][0])

                        if isinstance(rep,score.Note):
                            if openBeam:
                                if dur_dots[0][0]<8:
                                    openBeam=False
                                    parent = layer
                            elif dur_dots[0][0]>=8 and next_dur_dots[0][0]>=8 and (chord_i<len(chords)-1 and not insertElem_check(chords[chord_i+1][0].start.t, clef, keySig, timeSig) or len(dur_dots)>1 and not insertElem_check(rep.start.t+firstTempDur, clef, keySig, timeSig)):
                                parent = addChild(layer,"beam")
                                openBeam = True

                            if len(chordNotes)>1:
                                chord = addChild(parent,"chord")
                                setAttributes(chord,("dur",dur_dots[0][0]),("dots",dur_dots[0][1]))

                                for n in chordNotes:
                                    note=addChild(chord,"note")
                                    setAttributes(note,(xmlIdString,n.id),("pname",n.step.lower()),("oct",n.octave))
                            else:
                                note=addChild(parent,"note")
                                setAttributes(note,(xmlIdString,rep.id),("pname",rep.step.lower()),("oct",rep.octave),("dur",dur_dots[0][0]),("dots",dur_dots[0][1]))

                            # additional ties will have to work together with ties specified in partitura
                            if len(dur_dots)>1:
                                for n in chordNotes:
                                    ties[n.id]=[n.id]

                                for i in range(1,len(dur_dots)):
                                    if not openBeam and dur_dots[i][0]>=8:
                                        parent = addChild(layer,"beam")
                                        openBeam = True

                                    if len(chordNotes)>1:
                                        chord = addChild(parent,"chord")
                                        setAttributes(chord,("dur",dur_dots[i][0]),("dots",dur_dots[i][1]))

                                        for n in chordNotes:
                                            note=addChild(chord,"note")

                                            id = n.id+"-"+str(i)

                                            ties[n.id].append(id)

                                            setAttributes(note,(xmlIdString,id),("pname",n.step.lower()),("oct",n.octave))


                                    else:
                                        note=addChild(parent,"note")

                                        id = rep.id+"-"+str(i)

                                        ties[rep.id].append(id)

                                        setAttributes(note,(xmlIdString,id),("pname",n.step.lower()),("oct",n.octave),("dur",dur_dots[i][0]),("dots",dur_dots[i][1]))



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

                            rest = addChild(layer,"rest")

                            setAttributes(rest,(xmlIdString,rep.id),("dur",dur_dots[0][0]),("dots",dur_dots[0][1]))

                            if len(dur_dots)>1:
                                for i in range(1,len(dur_dots)):
                                    rest=addChild(layer,"rest")

                                    id = rep.id+str(i)

                                    setAttributes(rest,(xmlIdString,id),("dur",dur_dots[i][0]),("dots",dur_dots[i][1]))

                        if splitNotes!=None:
                            for sn in splitNotes:
                                sn.voice = rep.voice
                                sn.start = m.end
                                sn.end = score.TimePoint(rep.start.t+rep.duration)

                                if s in notes_nextMeasure_perStaff.keys():
                                    notes_nextMeasure_perStaff[s].append(sn)
                                else:
                                    notes_nextMeasure_perStaff[s]=[sn]
                    # for now all notes are beamed, however some rules should be obeyed there, see Note Beaming and Grouping


                    ties_perStaff_perVoice[voice]=ties

                ties_perStaff[s]=ties_perStaff_perVoice



        # staffs should probably be created in order


        create_staff(notes_withinMeasure_perStaff, ties_perStaff, notes_nextMeasure_perStaff)


        notes_lastMeasure_perStaff_keys = notes_lastMeasure_perStaff_keys.difference(staff_intersection_keys)
        notes_lastMeasure_perStaff = {k:v for k,v in notes_lastMeasure_perStaff.items() if k in notes_lastMeasure_perStaff_keys}

    create_staff(notes_lastMeasure_perStaff, ties_perStaff, notes_nextMeasure_perStaff)
    notes_lastMeasure_perStaff = notes_nextMeasure_perStaff

    for p in parts:
        for slur in p.iter_all(score.Slur, m.start, m.end):
            s = addChild(measure,"slur")
            setAttributes(s, ("staff",slur.start_note.staff), ("startid","#"+slur.start_note.id), ("endid","#"+slur.end_note.id))


    for s,tps in ties_perStaff.items():

        for v,tpspv in tps.items():

            for ties in tpspv.values():

                for i in range(len(ties)-1):
                    tie = addChild(measure, "tie")

                    setAttributes(tie, ("staff",s), ("startid","#"+ties[i]), ("endid","#"+ties[i+1]))







(etree.ElementTree(mei)).write("testResult.mei",pretty_print=True)

#print(etree.tostring(mei,pretty_print=True))
