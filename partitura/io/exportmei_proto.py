import partitura
import partitura.score as score
from lxml import etree
from partitura.utils.generic import partition
from partitura.utils.music import estimate_symbolic_duration
import sys


part = score.Part("P0","Test")

# part.set_quarter_duration(0,2)
# part.add(score.KeySignature(0,"major"),start=0)
# part.add(score.TimeSignature(4,4),start=0)
#
# part.add(score.Clef(sign="G",line=2, octave_change=0, number=1),start=0)
# part.add(score.Clef(sign="F",line=4, octave_change=0, number=2),start=0)
#
#
#
# part.add(score.Note(id="n0s2",step="C",octave=4, staff=1, voice=1),start=0,end=5)
# part.add(score.Note(id="n2s2",step="G",octave=4, staff=1, voice=1),start=5,end=6)
# part.add(score.Note(id="n3s2",step="E",octave=4, staff=1, voice=1),start=6,end=7)
# part.add(score.Note(id="n4s2",step="F",octave=4, staff=1, voice=1),start=7,end=9)
# part.add(score.Note(id="n5s2",step="D",octave=4, staff=1, voice=1),start=9,end=10)
# part.add(score.Note(id="n6s2",step="A",octave=4, staff=1, voice=1),start=10,end=16)
#
# part.add(score.Note(id="n0s22",step="E",octave=4, staff=1, voice=1),start=0,end=5)
# part.add(score.Note(id="n2s22",step="B",octave=4, staff=1, voice=1),start=5,end=6)
# part.add(score.Note(id="n3s22",step="G",octave=4, staff=1, voice=1),start=6,end=7)
# part.add(score.Note(id="n4s22",step="A",octave=4, staff=1, voice=1),start=7,end=9)
# part.add(score.Note(id="n5s22",step="F",octave=4, staff=1, voice=1),start=9,end=10)
# part.add(score.Note(id="n6s22",step="C",octave=5, staff=1, voice=1),start=10,end=16)
#
# part.add(score.Note(id="n0",step="C",octave=2, staff=2, voice=1),start=0,end=3)
# part.add(score.Note(id="n2",step="E",octave=2, staff=2, voice=1),start=3,end=6)
# part.add(score.Note(id="n3",step="G",octave=2, staff=2, voice=1),start=6,end=7)
# part.add(score.Note(id="n4",step="D",octave=2, staff=2, voice=1),start=7,end=10)
# part.add(score.Note(id="n5",step="F",octave=2, staff=2, voice=1),start=10,end=13)
# part.add(score.Note(id="n6",step="A",octave=2, staff=2, voice=1),start=13,end=16)
# score.add_measures(part)
# test case blues lick
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

part = partitura.load_musicxml("../../tests/data_examples/Three-Part_Invention_No_13_(fragment).xml", force_note_ids=True)

qd=part.quarter_durations()[0][1]

part.add(score.Clef(sign="F",line=4, octave_change=0, number=2), start=int(qd*(2+1/4)))
part.add(score.KeySignature(-3,"minor"),start=int(qd*(2+1/4)))


# create MEI file from this ;D

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

keySig = next(part.iter_all(score.KeySignature),None)

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

if keySig!=None:
    fifths, mode, pname = attribsOf_keySig(keySig)

    setAttributes(scoreDef,("key.sig",fifths),("key.mode", mode),("key.pname",pname))

timeSig = next(part.iter_all(score.TimeSignature),None)

if timeSig!=None:
    setAttributes(scoreDef,("meter.count",timeSig.beats),("meter.unit",timeSig.beat_type))

section = addChild(mei_score,"section")

# might want to count staff numbers during processing and update staffGrp if count isn't consistent with clefs
staffGrp = addChild(scoreDef,"staffGrp")

def partition_handleNone(func, iter, partitionAttrib):
    p = partition(func,iter)

    if None in p.keys():
        print("PARTITION ERROR: some elements of set do not have partition attribute \""+partitionAttrib+"\"")
        sys.exit()

    return p

clefs = partition_handleNone(lambda c:c.number, part.iter_all(score.Clef), "number")

if len(clefs)==0:
    staffDef = addChild(staffGrp,"staffDef")
else:
    for c in clefs.values():
        clef = c[0]

        for i in range(1,len(c)):
            if c[i].start.t < clef.start.t:
                clef = c[i]

        if clef.number == None:
            print("Encountered clef which isn't assigned to a staff, please assign following clef:")
            print(clef)
            sys.exit()

        staffDef = addChild(staffGrp,"staffDef")
        setAttributes(staffDef,("n",clef.number),("lines",5),("clef.shape",clef.sign),("clef.line",clef.line))








measure_counter = 0

notes_nextMeasure_perStaff = {}

for m in part.iter_all(score.Measure):
    clefs_withinMeasure_perStaff = partition_handleNone(lambda c:c.number, part.iter_all(score.Clef, m.start, m.end),"number")
    keySigs_withinMeasure = list(part.iter_all(score.KeySignature, m.start, m.end))

    measure=addChild(section,"measure")
    setAttributes(measure,("n",m.number))

    quarterDur = m.start.quarter

    notes_withinMeasure_perStaff = partition_handleNone(lambda n:n.staff, part.iter_all(score.GenericNote, m.start, m.end, include_subclasses=True), "staff")

    notes_nextMeasure_perStaff_keys=set(notes_nextMeasure_perStaff.keys())
    notes_withinMeasure_perStaff_keys=set(notes_withinMeasure_perStaff.keys())

    staff_intersection_keys = notes_nextMeasure_perStaff_keys.intersection(notes_withinMeasure_perStaff_keys)

    for s in staff_intersection_keys:
        notes_withinMeasure_perStaff[s].extend(notes_nextMeasure_perStaff[s])

    def create_staff(notes_perStaff, ties_perStaff):
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
                            print("In staff "+str(s)+",")
                            print("in measure "+str(m.number)+",")
                            print("for voice "+str(voice)+",")
                            print("2 notes start at time "+str(n.start.t)+",")
                            print("but have different durations, namely "+n.id+" has duration "+str(n.duration)+" and "+rep.id+" has duration "+str(rep.duration))
                            print("change to same duration for a chord or change voice of one of the notes for something else")
                            sys.exit()
                    else:
                        chords.append([n])

                def calc_dur_dots_splitNotes(n):
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

                    return dur_dots,splitNotes



                openBeam=False
                parent = layer

                next_dur_dots, next_splitNotes = calc_dur_dots_splitNotes(chords[0][0])

                clef_i=0
                clef = None
                if s in clefs_withinMeasure_perStaff.keys():
                    if measure_counter==0:
                        if len(clefs_withinMeasure)>1:
                            clef = clefs_withinMeasure[1]
                    else:
                        clef = clefs_withinMeasure[0]

                keySig_i=0
                keySig = None

                if len(keySigs_withinMeasure)>0:
                    if measure_counter==0:
                        if len(keySigs_withinMeasure)>1:
                            keySig = keySigs_withinMeasure[1]
                    else:
                        keySig = keySigs_withinMeasure[0]

                def insertElem_check(elem, note):
                    return elem!=None and elem.start.t<=note.start.t

                def insertElem(elem, note, elemName, attribNames, attribValsOf_elem, cursor, elemContainer):
                    ob = openBeam
                    p = parent
                    if insertElem_check(elem,note):
                        # note should also be split according to keysig or clef etc insertion time, right now only beaming is disrupted
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
                    dur_dots,splitNotes = next_dur_dots, next_splitNotes

                    openBeam, parent, clef_i, clef = insertElem(clef, rep, "clef", ["shape","line"], lambda c:(c.sign,c.line), clef_i, clefs_withinMeasure)

                    openBeam, parent, keySig_i, keySig = insertElem(keySig, rep, "keySig", ["sig","mode","pname"], attribsOf_keySig, keySig_i, keySigs_withinMeasure)


                    # hack right now, don't need to check every iteration, good time to factor out inside of loop
                    if chord_i < len(chords)-1:
                        next_dur_dots, next_splitNotes = calc_dur_dots_splitNotes(chords[chord_i+1][0])

                    if isinstance(rep,score.Note):
                        if openBeam:
                            if dur_dots[0][0]<8:
                                openBeam=False
                                parent = layer
                        elif dur_dots[0][0]>=8 and (next_dur_dots[0][0]>=8 and chord_i<len(chords)-1 or len(dur_dots)>1) and not insertElem_check(clef, chords[chord_i+1][0]) and not insertElem_check(keySig, chords[chord_i+1][0]):
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

    ties_perStaff = {}

    # staffs should probably be created in order

    create_staff(notes_withinMeasure_perStaff, ties_perStaff)


    notes_nextMeasure_perStaff_keys = notes_nextMeasure_perStaff_keys.difference(staff_intersection_keys)

    create_staff({k:v for k,v in notes_nextMeasure_perStaff.items() if k in notes_nextMeasure_perStaff_keys}, ties_perStaff)



    for slur in part.iter_all(score.Slur, m.start, m.end):
        s = addChild(measure,"slur")
        setAttributes(s, ("staff",slur.start_note.staff), ("startid","#"+slur.start_note.id), ("endid","#"+slur.end_note.id))


    for s,tps in ties_perStaff.items():

        for v,tpspv in tps.items():

            for ties in tpspv.values():

                for i in range(len(ties)-1):
                    tie = addChild(measure, "tie")

                    setAttributes(tie, ("staff",s), ("startid","#"+ties[i]), ("endid","#"+ties[i+1]))

    measure_counter+=1






(etree.ElementTree(mei)).write("testResult.mei",pretty_print=True)

#print(etree.tostring(mei,pretty_print=True))
#partitura.render(part)