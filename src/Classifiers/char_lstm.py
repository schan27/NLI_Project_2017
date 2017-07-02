import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from keras.layers import LSTM, Lambda
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import re
import keras.callbacks
import sys
import os
import argparse

__author__ = 'Maryam'


### ATTRIBUTES

lang = {'ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR'}
langdict = {'ARA': 0, 'CHI': 1, 'FRE': 2, 'GER': 3, 'HIN': 4, 'ITA': 5,
            'JPN': 6, 'KOR': 7, 'SPA': 8, 'TEL': 9, 'TUR': 10}

MAXLEN = 150
MAX_SENTENCES = 15
filter_length = [5, 3, 3]
nb_filter = [75, 196, 256]
pool_length = 2

### FUNCTIONS
def binarize(x, sz=66):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def binarize_outshape(in_shape, sz = 66):
    return in_shape[0], in_shape[1], 66


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


def striphtml(html):
    p = re.compile(r'<.*?>')
    return p.sub('', html)


def read_data (df):

    docs = []
    sentences = []
    lang1 = []

    for cont, l1 in zip(df.essay, df.label):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',
                             clean(striphtml(cont)))
        sentences = [sent.lower() for sent in sentences]
        docs.append(sentences)
        lang1.append(l1)

    return docs, sentences, lang1


# record history of training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))



if __name__ == '__main__':
    total = len(sys.argv)
    cmdargs = str(sys.argv) #??

    print ("Script name: %s" % str(sys.argv[0]))

    parser = argparse.ArgumentParser(description='This is a demo script by nixCraft.')
    parser.add_argument('-base', '--baseFolder',
                        help='base folder \n ../../nli-shared-task-2017/data/essays/',
                        required=True)
    parser.add_argument('-chp', '--checkpointPath',
                        help='checkpoint path if exists', required=False)
    args = parser.parse_args()


    checkpoint = None

    if os.path.exists(str(args.checkpointPath)):
        print("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])


    basedir = args.baseFolder

    trainpath = os.path.join(basedir, 'train', 'raw_csv', 'train' + '.csv')
    devpath = os.path.join(basedir, 'dev', 'raw_csv', 'dev' + '.csv')

    traindf = pd.read_csv(trainpath)
    devdf = pd.read_csv(devpath)

    docs_train, sentences_train, l1_train = read_data(traindf)
    docs_dev, sentences_dev, l1_dev = read_data(devdf)


### make set of chars
    txt = ''
    numsent_train = []
    for doc in docs_train:
        numsent_train.append(len(doc))
        for s in doc:
            txt += s

    numsent_dev =[]
    for doc in docs_dev:
        numsent_dev.append(len(doc))
        for s in doc:
            txt +=s

    chars = set(txt)

    print('total chars:', len(chars))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print('Sample doc{}'.format(docs_train[100]))

### put train data in matrix shape
    x_train = np.ones((len(docs_train), MAX_SENTENCES, MAXLEN), dtype=np.int64) * -1
    y_train = np.array(to_categorical(l1_train))

    print(to_categorical(l1_train))
    print(y_train[:10])

    for i, doc in enumerate(docs_train):
        for j , sent in enumerate(doc):
            if j < MAX_SENTENCES:
                for t, char in enumerate(sent[-MAXLEN:]):
                    x_train[i, j, (MAXLEN -1 - t)] = char_indices[char]

    print('Sample chars in X:{}'.format(x_train[100, 2]))
    print('y:{}'.format(y_train[12]))

### put test data into matrix shape

    x_dev = np.ones((len(docs_dev), MAX_SENTENCES, MAXLEN), dtype=np.int64) * -1
    y_dev = np.array(to_categorical(l1_dev))

    print(to_categorical(l1_dev))
    print(y_dev[:10])

    for i, doc in enumerate(docs_dev):
        for j, sent in enumerate(doc):
            if j < MAX_SENTENCES:
                for t, char in enumerate(sent[-MAXLEN:]):
                    x_dev[i, j, (MAXLEN - 1 - t)] = char_indices[char]

    print('Sample chars in X:{}'.format(x_dev[100, 2]))
    print('y:{}'.format(y_dev[12]))

    ### could shuffle data but I didn't


    #### model design

    # document input
    document = Input(shape=(MAX_SENTENCES, MAXLEN), dtype='int64')
    # sentence input
    in_sentence = Input(shape=(MAXLEN,), dtype='int64')
    # char indices to one hot matrix, 1D sequence to 2D
    embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
    # embedded: encodes sentence
    for i in range(len(nb_filter)):
        embedded = Conv1D(filters=nb_filter[i],
                          kernel_size=filter_length[i],
                          padding='valid',
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          strides=1)(embedded)

        embedded = Dropout(0.1)(embedded)
        embedded = MaxPooling1D(pool_size=pool_length)(embedded)

    bi_lstm_sent = \
        Bidirectional(LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(
            embedded)

    # sent_encode = merge([forward_sent, backward_sent], mode='concat', concat_axis=-1)
    sent_encode = Dropout(0.3)(bi_lstm_sent)
    # sentence encoder
    encoder = Model(inputs=in_sentence, outputs=sent_encode)
    encoder.summary()

    encoded = TimeDistributed(encoder)(document)
    # encoded: sentences to bi-lstm for document encoding
    b_lstm_doc = \
        Bidirectional(LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(
            encoded)

    output = Dropout(0.3)(b_lstm_doc)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(11, activation='softmax')(output)

    model = Model(inputs=document, outputs=output)

    model.summary()
### model run


if checkpoint:
    model.load_weights(checkpoint)

file_name = os.path.basename(sys.argv[0]).split('.')[0]
check_cb = keras.callbacks.ModelCheckpoint('checkpoints/' + file_name + '.{epoch:02d}-{val_loss:.4f}-{val_acc:.5f}.hdf5',
                                           monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
history = LossHistory()
model.compile(loss='categorical_crossentropy', optimizer='rmspro', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=10,
          epochs=5, shuffle=True, callbacks=[earlystop_cb, check_cb, history])


# just showing access to the history object
print(history.losses)
print(history.accuracies)










