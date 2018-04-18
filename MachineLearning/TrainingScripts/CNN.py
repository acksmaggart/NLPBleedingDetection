import re
import pandas as pd
import os
import ParseTexts
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Input
from keras.models import Model
from keras.callbacks import TensorBoard
import sklearn
from sklearn.metrics import precision_recall_fscore_support
import os


MAX_NUM_WORDS = 2000
VALIDATION_FRACTION = .2
THIS_SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
GLOVE_EMBEDDINGS_PATH = 'path/to/glove/embeddings'
EMBEDDING_DIM = 100
GOLD_STANDARD_PATH = "path/to/gold/standard/document/classes"
CORPUS_TRAIN_PATH = "path/to/training/notes"
SERIALIZED_MODEL_PATH = "path/to/serialized/model" ## this is the location where the trained model will be saved.

# Metrics
def specificity(true, predicted):
    numTrueNegatives = np.sum(np.where(true == 0, 1., 0.))
    numAgreedNegatives = np.sum(((predicted - 1) * -1) * ((true - 1) * -1))
    return float(numAgreedNegatives) / float(numTrueNegatives)

def NPV(true, predicted):
    true = (true - 1) * -1
    predicted = (predicted - 1) * -1
    return sklearn.metrics.precision_score(true, predicted)



# Helper Functions
def convertTextClassesToIntClasses(rawTextLabels):
    classes = []
    processedLabels = np.zeros(len(rawTextLabels))
    for index, label in enumerate(rawTextLabels):
        if label not in classes:
            classes.append(label)
        processedLabels[index] = classes.index(label)

    return processedLabels, len(classes)

def getNotesAndClasses(corpusPath, truthPath, balanceClasses=False):
    truthData = pd.read_csv(truthPath, dtype={"notes": np.str, "classes": np.int}, delimiter='\t',
                            header=None).as_matrix()

    noteNames = truthData[:, 0].astype(str)
    noteClasses = truthData[:, 1]

    if balanceClasses:
        np.random.seed(8229)
        noteNames = np.array(noteNames)
        posIndices = np.where(noteClasses == 1)[0]
        negIndices = np.where(noteClasses == 0)[0]
        posNotes = noteNames[posIndices]
        negNotes = noteNames[negIndices]
        assert len(posNotes) + len(negNotes) == len(noteNames)

        selectedNegNotes = np.random.choice(negNotes, size=len(posNotes), replace=False)
        allNotes = np.concatenate((posNotes, selectedNegNotes), axis=0)
        labels = np.concatenate((np.ones(len(posNotes)), np.zeros(len(selectedNegNotes))), axis=0)

        noteNames = allNotes
        noteClasses = labels

    noteBodies = []

    for name in noteNames:
        with open(corpusPath + name + ".txt") as inFile:
            noteBodies.append(inFile.read().lower())
    return np.array(noteBodies), noteClasses.astype(int)


# Begin data prep and training.
cleanTexts, labels = getNotesAndClasses(CORPUS_TRAIN_PATH, GOLD_STANDARD_PATH, balanceClasses=False)
numLabels = 2

# Start fitting the convolutional model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(cleanTexts)
sequences = tokenizer.texts_to_sequences(cleanTexts)

wordIndex = tokenizer.word_index
print("Found %i words." % len(wordIndex))


data = pad_sequences(sequences, maxlen=MAX_NUM_WORDS)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
numValidationSamples = int(VALIDATION_FRACTION * len(indices))
xTrain = data[: -numValidationSamples]
yTrain = labels[: -numValidationSamples]
xTest = data[-numValidationSamples:]
yTest = labels[-numValidationSamples:]

embeddings_index = {}
f = open(GLOVE_EMBEDDINGS_PATH)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
notFoundCount = 0
for word, i in wordIndex.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        notFoundCount += 1
print "Not Found: %i" % notFoundCount

embedding_layer = Embedding(len(wordIndex) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_NUM_WORDS,
                            trainable=False)


# Build the network:
sequence_input = Input(shape=(MAX_NUM_WORDS,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(xTrain, yTrain, validation_data=(xTest, yTest),
          epochs=4, batch_size=8)
model.save(SERIALIZED_MODEL_PATH)

predictions = model.predict(xTest)
reshapedPreds = np.reshape(np.where(predictions > .5, 1, 0), (predictions.shape[0],))

precision, recall, fscore, _ = precision_recall_fscore_support(yTest, reshapedPreds, average="binary")
npv = NPV(yTest, reshapedPreds)
specificity_score = specificity(yTest, reshapedPreds)

print"F-Score: %.3f\nPrecision: %.3f\nRecall (Sensitivity): %.3f\nSpecificity: %.3f\nNPV: %.3f" \
    % (fscore, precision, recall, specificity_score, npv)
