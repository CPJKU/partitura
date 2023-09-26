import partitura as pt

score = pt.load_mei(r"C:\Users\fosca\Desktop\JKU\partitura\tests\data\mei\example_noMeasures_noBeams.mei")
print(score.parts)
