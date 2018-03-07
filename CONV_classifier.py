from __future__ import print_function

import os
import codecs
import gensim
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, model_from_json

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector
print('Loading model...')
word_model = gensim.models.KeyedVectors.load_word2vec_format("data\\W2V_vectors.bin", binary=True)
model_size = len(word_model.wv.vocab)
embedding_weights = np.zeros((model_size, EMBEDDING_DIM))
print('Indexing word vectors...')
embeddings_index = {}
for inx in range(model_size):
    word = word_model.wv.index2word[inx]
    embeddings_index[word] = word_model.wv[word]

# second, prepare text samples and their labels
print('Processing text dataset')
texts = []  # list of text samples
labels_index = dict()  # dictionary mapping label name to numeric id
labels = []  # list of label ids
S = u'абвгдежзийклмнопрстуфхцчшщыъьэюя'
dir_inx = 0

INPUT_DIR = "data\\TEXTS"
#INPUT_DIR = "data\\MORHEME_TEXTS"

for dir in os.listdir(INPUT_DIR):
    labels_index[os.path.dirname(dir)] = dir_inx
    for file in os.listdir(INPUT_DIR + "\\" + dir):
        train_text = ""
        with codecs.open(INPUT_DIR + "\\" + dir + "\\" + file, "r", encoding="utf-8") as input_file:
            for line in input_file:
                line = line.replace("-", " ")
                line = line.replace(u"—", " ")
                line = line.replace(u"–", " ")
                word_list = line.split(" ")
                for word in word_list:
                    new_word = ''
                    letters_list = list(word.lower())
                    for letter in letters_list:
                        if letter in S:
                            new_word += letter

                    new_word = new_word.strip()
                    if (len(new_word) and (new_word in word_model.wv.vocab)):
                        train_text = train_text + new_word + " "

        train_text = train_text.strip()
        if len(train_text):
            texts.append(train_text)
            labels.append(labels_index[os.path.dirname(dir)])
    dir_inx += 1

print('Found %s texts.' % len(texts))
# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
#num_words = len(word_index)
embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(dir_inx, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


#keras.callbacks.ModelCheckpoint("model.{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=25)

model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          validation_data=(x_val, y_val))

#check result

scores = model.evaluate(x_val, y_val)
print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))
result = model.predict(x_val)
print(np.argmax(result, axis=1))


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


loaded_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

#check result
scores = loaded_model.evaluate(x_val, y_val)
print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))
result = loaded_model.predict(x_val)
print(np.argmax(result, axis=1))