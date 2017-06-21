import os
import re
import numpy as np
import pandas as pd
import sys
import pickle
import ast
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
# from os import path, mkdir, listdir


BASE_DIR = '../../nli-shared-task-2017/data/essays/'
EMBEDDING_FILE = '../lib/word2vec/' + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = ''
TEST_DATA_FILE = ''
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
NB_CLASSES = 11
BATCHSIZE = 2048
NB_EPOCHS = 300


num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \

      rate_drop_dense)

def model_0():

    #################
    # load data
    #################
    df_train = pd.read_csv(TRAIN_DATA_FILE)
    x = df_train.sent.tolist()
    y = df_train.label.tolist()
    y = [ast.literal_eval(li) for li in y]

    # print(type(y), len(y), type(y[3]), y[3])
    # lb = np.array(y)
    # print(lb.shape, type(lb[3]))
    # exit()
    df_test = pd.read_csv(TEST_DATA_FILE)
    x_test = df_test.sent.tolist()
    y_test = df_test.label.tolist()
    y_test = [ast.literal_eval(li) for li in y_test]

    tokenizer = Tokenizer(num_words= MAX_NB_WORDS, lower=False)
    print(type(x), len(x), x[:10])
    print(len(x+x_test))
    tokenizer.fit_on_texts(x + x_test)

    seq = tokenizer.texts_to_sequences(x)
    seq_test = tokenizer.texts_to_sequences(x_test)

    word_index = tokenizer.word_index
    print('found %s unique tokens' % len(word_index))

    data = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

    labels = np.array(y)

    print('shape of data: ', data.shape)
    print('shape of labels: ', labels.shape)

    data_test = pad_sequences(seq_test, maxlen=MAX_SEQUENCE_LENGTH)
    labels_test = np.array(y_test)
    print('shape of data: ', data_test.shape)
    print('shape of labels: ', labels_test.shape)


    ########################################
    ## index word vectors
    ########################################
    # print('Indexing word vectors')
    #
    # word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
    #         binary=True)
    # print('Found %s word vectors of word2vec' % len(word2vec.vocab))


    ###########################
    ##  prepair embedings
    ###########################

    print('prepairing embeddings')
    nb_words = min(MAX_NB_WORDS, len(word_index))+1

    embedding_matrix =  np.zeros((nb_words, EMBEDDING_DIM))
    # for word, i in word_index.items():
    #       if word in word2vec.vocab:
    #             embedding_matrix[i] = word2vec.word_vec(word)
    #       else:
    #             print('no embedding for word: ', word )
    # print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    #
    # pickle.dump(embedding_matrix, open('embding_matrix.pkl', 'wb'))
    # del word2vec

    embedding_matrix = pickle.load(open('embding_matrix.pkl', 'rb'))

    perm = np.random.permutation(len(data))
    idx_train = perm[:int(len(data)*(1-VALIDATION_SPLIT))]
    idx_val = perm[int(len(data)*(1-VALIDATION_SPLIT)):]

    data_train = data[idx_train]
    labels_train = labels[idx_train]

    data_val = data[idx_val]
    labels_val = labels[idx_val]


    ###################################
    ## model structure
    ###################################
    print(STAMP)
    model = Sequential()
 
    model.add(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    model.add(Dropout(rate_drop_dense))
    model.add(BatchNormalization())
    model.add(Dense(NB_CLASSES, activation='softmax'))
    print('model.summary:', model.summary())

    # embedding_layer = Embedding(nb_words, EMBEDDING_DIM,
    #                             weights=[embedding_matrix],
    #                             input_length= MAX_SEQUENCE_LENGTH,
    #                             trainable=False)
    # lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    #
    # seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # embedded_seq = embedding_layer(seq_input)
    # lstm_out = lstm_layer(embedded_seq)
    #
    # lstm_out = Dropout(rate_drop_dense)(lstm_out)
    # lstm_out = BatchNormalization()(lstm_out)
    #
    # pred = Dense(NB_CLASSES, activation='softmax')(lstm_out)


    ############################################
    ## train model
    ############################################

    # model = Model(inputs=seq_input, outputs=pred)

    myadam = adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=myadam,  # 'adam',
                  metrics=['accuracy'])

    print(STAMP)
    print(labels_train[:10])
    print('train: ', data_train.shape, labels_train.shape)
    print('val: ', data_val.shape)
    early_stopping =EarlyStopping(monitor='val_loss', patience=10)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    hist = model.fit(data_train, labels_train, \
            validation_data=(data_val, labels_val), \
            epochs=NB_EPOCHS, batch_size=BATCHSIZE, shuffle=True, \
            callbacks=[early_stopping, model_checkpoint])

    # model.load_weights(bst_model_path)
    # bst_val_score = min(hist.history['val_loss'])# WHY MIN?
    fold_acc = hist.history['val_acc']
    val_acc = model.evaluate(data_test, labels_test, batch_size=128, verbose=1)

    print('Start making the submission before fine-tuning')

    preds = model.predict(data_test, batch_size=8192, verbose=1)

    submission = pd.DataFrame({'test_id':labels, 'is_duplicate':preds.ravel()})
    submission.to_csv('%.4f_'%(val_acc)+'%.4f_'%(fold_acc)+STAMP+'.csv', index=False)

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts



if __name__ == '__main__':

    from sys import  argv
    myargs = getopts(argv)
    BASE_DIR  = myargs['-base']
    TRAIN_DATA_FILE = os.path.join(BASE_DIR+ 'train/lstm_in/', 'train_corrected_cleaned.csv')
    TEST_DATA_FILE = os.path.join(BASE_DIR+ 'dev/lstm_in/', 'dev_corrected_cleaned.csv')
    print(TRAIN_DATA_FILE)
    df_train = pd.read_csv(TRAIN_DATA_FILE)
    model_0()

