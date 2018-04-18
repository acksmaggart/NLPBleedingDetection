"""Compare PyConText annotation to gold standard using all the annotated notes every time. Exclude documents that only
have their document class adjudicated. Remove the DOC CLASS annotation from all the eHost-annotated documents to remove
noise in the comparison."""

from eHostess.PyConTextInterface.PyConText import PyConTextInterface
import pyConTextNLP.itemData as itemData
from eHostess.eHostInterface.KnowtatorReader import KnowtatorReader
import eHostess.PyConTextInterface.SentenceSplitters.PyConTextBuiltinSplitter as BuiltinSplitter
import eHostess.PyConTextInterface.SentenceSplitters.TargetSpanSplitter as TargetSpanSplitter
from eHostess.Analysis.DocumentComparison import Comparison
from eHostess.Analysis.Output import ConvertComparisonsToTSV
import eHostess.Analysis.Metrics as Metrics
from eHostess.Annotations.Document import ClassifiedDocument
import numpy as np

adjudicatedDirectories = ['a list of paths, each to a directory containing annotated notes. Typically when using an annotation tool like eHost notes will be divided into several batches, 
                            this is a list of paths to those batches.']
corpusDirectories = ['for our project the raw training documents were split up into batches, this is a list of directories containing those batches']
PATH_TO_TARGETS = "./targets.tsv"
PATH_TO_MODIFIERS = "./modifiers.tsv"

print "Parsing adjudicated docs..."
adjudicatorDocs = KnowtatorReader.parseMultipleKnowtatorFiles(adjudicatedDirectories)
if len(adjudicatorDocs) != 30 * len(adjudicatedDirectories):
    raise RuntimeError("There should be %i annotated eHost documents. But there are %i." % (4 * len(adjudicatedDirectories), len(adjudicatorDocs)))

targets = itemData.instantiateFromCSVtoitemData(PATH_TO_TARGETS)
pyConTextInput = TargetSpanSplitter.splitSentencesMultipleDocuments(corpusDirectories, targets, 8, 8)

pyConTextDocs = PyConTextInterface.PerformAnnotation(pyConTextInput)
if len(pyConTextDocs) != 30 * len(corpusDirectories):
    raise RuntimeError("There should be %i annotated pyConText documents. But there are %i." % (4 * len(corpusDirectories), len(pyConTextDocs)))

classifiedAdjudicatorDocs = []
for doc in adjudicatorDocs:
    annotations = doc.annotations
    annotationsToKeep = []
    currentDocumentClass = ''
    for annotation in annotations:
        if annotation.annotationClass == 'doc_classification':
            currentDocumentClass = annotation.attributes["present_or_absent"]
        else:
            annotationsToKeep.append(annotation)

    classifiedAdjudicatorDocs.append(ClassifiedDocument(doc.documentName, doc.annotationGroup, annotationsToKeep, doc.numberOfCharacters, currentDocumentClass, doc.adjudicationStatus))

# Remove notes where only the doc class has been adjudicated.
filteredAdjudicated = []
filteredPyConText = []
classifiedAdjudicatorDocs.sort(key=lambda x: x.documentName)
pyConTextDocs.sort(key=lambda x: x.documentName)

for index in range(len(classifiedAdjudicatorDocs)):
    if len(classifiedAdjudicatorDocs[index].annotations) != 0:
        filteredAdjudicated.append(classifiedAdjudicatorDocs[index])
        filteredPyConText.append(pyConTextDocs[index])

comparisons = Comparison.CompareDocumentBatches(filteredAdjudicated, filteredPyConText, equivalentClasses=[["bleeding_present"], ["bleeding_absent", "bleeding_hypothetical", "bleeding_historical"]], equivalentAttributes=False, countNoOverlapAsMatch=['bleeding_absent', 'bleeding_hypothetical', 'bleeding_historical'])
precision, recall, fscore, agreement = Metrics.CalculateRecallPrecisionFScoreAndAgreement(comparisons)

#ConvertComparisonsToTSV(comparisons, '/users/shah/Box Sync/MIMC_v2/Annotation/Automated/PyConTextIteration/ComparisonOutput/Round5Builtin.txt')

print precision, recall, fscore, agreement

# Calculate agreement separately for positive and negative annotations
negatives = []
positives = []
for comparison in comparisons:
    if comparison.annotation1:
        trueClass = comparison.annotation1.annotationClass
    else:
        trueClass = "bleeding_absent"
    if comparison.annotation2:
        predictedClass = comparison.annotation2.annotationClass
    else:
        predictedClass = "bleeding_absent"

    if trueClass == "bleeding_present":
        if predictedClass == "bleeding_present":
            positives.append(1)
        else:
            positives.append(0)
    else:
        if predictedClass == "bleeding_present":
            negatives.append(0)
        else:
            negatives.append(1)

positives = np.array(positives)
negatives = np.array(negatives)

print "Positive agreement: %.4f" % (float(np.sum(positives)) / float(len(positives)))
print "Negative agreement: %.4f" % (float(np.sum(negatives)) / float(len(negatives)))
