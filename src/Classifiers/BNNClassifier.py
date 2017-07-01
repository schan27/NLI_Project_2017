from .ClassifierBase import ClassifierBase, LanguageGroup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_extraction.text import TfidfTransformer
from src.util.Misc import max_index, select_feature
from keras.models import model_from_json
from sklearn.pipeline import Pipeline
import numpy as np
import tensorflow as tf
import os

FEATURE_COUNT = 'all'
FEATURE = "original"
N_CLASSES = 11
LOAD_MODEL = False


class BNN:
    def __init__(self, pout=False):
        super(BNN,self).__init__()
        one = _BNN(ngram_range=(1, 1), lower=True, binary=False)
        #one.add_pipe(Pipeline([('cv',CountVectorizer(ngram_range=(1,1),binary=False)),
        #                       ('idf', TfidfTransformer(norm='l2')),
        #                       ('kb',SelectKBest(k=FEATURE_COUNT))]))
        one.add_pipe(Pipeline([('cv',CountVectorizer(ngram_range=(2,2),binary=True)),
                               #('idf',TfidfTransformer(norm='l2')),
                               ('kb',SelectKBest(k=100000))]))
        one.add_pipe(Pipeline([('cv', CountVectorizer(ngram_range=(3, 3), binary=False)),
                               #('idf', TfidfTransformer(norm='l2')),
                               ('kb', SelectKBest(k=100000))]))

        two = _BNN()
        two.add_pipe(Pipeline([('cv', CountVectorizer(ngram_range=(1, 4), binary=True,lowercase=False)),
                               #('idf', TfidfTransformer(norm='l2')),
                               ('kb', SelectKBest(k=100000))]))

        three = _BNN()
        three.add_pipe(Pipeline([('cv', CountVectorizer(ngram_range=(5, 5), binary=True, analyzer="char")),
                               #('idf', TfidfTransformer(norm='l2')),
                               ('kb', SelectKBest(k=100000))]))

        four = _BNN()
        four.add_pipe(Pipeline([('cv', CountVectorizer(ngram_range=(5, 5), binary=True, analyzer="char_wb")),
                                 # ('idf', TfidfTransformer(norm='l2')),
                                 ('kb', SelectKBest(k='all'))]))

        self.network_list = [one,two,three,four]#_BNN(binary=True,analyzer="char_wb",feature_count='all'), _BNN(binary=True),_BNN(binary=False),_BNN(binary=True, analyzer='char'),
                             #_BNN(binary=False, analyzer='char')]
        self.prob_output = pout

    def preprocess(self,data):
        return data

    def load_models(self):
        dirname = os.path.dirname(__file__)
        dirname += "models/"
        i = 0
        for i in range(0,len(self.network_list)):
            model_from_json(dirname + "1.json")

            i += 1



    def train(self,training_data):
        if len(training_data) == 1:
            feature_data = []
            feature_data.append(select_feature(training_data[0][1],training_data[0][0],"original"))
            feature_data.append(select_feature(training_data[0][1], training_data[0][0], "lemmas"))

            i = 0
            for network in self.network_list:
                print("Training Network: ", i + 1, " of: ", len(self.network_list))
                network.train(None, feature_data)
                i += 1
        else:
            i = 0
            for network in self.network_list:
                print("Training Network: ", i+1, " of: ", len(self.network_list))
                network.train(training_data)
                i += 1

    def classify(self,testing_data):
        network_results = []
        label_list = []

        if len(testing_data) == 1:

            feature_data = []
            feature_data.append(select_feature(testing_data[0][1], testing_data[0][0], "original"))
            feature_data.append(select_feature(testing_data[0][1], testing_data[0][0], "lemmas"))

            for data in feature_data[0]:
                label_list.append(data[0])

            i = 0
            for network in self.network_list:
                network_results.append(network.classify(None, feature_data))
                i += 1
        else :
            label_list = []
            for test in testing_data:
                label_list.append(test[0])

            network_results = []
            for network in self.network_list:
                network_results.append(network.classify(testing_data))

        if self.prob_output:
            return network_results
        else:
            output = []
            for i in range(0, len(network_results[0])):
                output_merge = network_results[0][i]
                for z in range(1, len(network_results)):
                    for k in range(0, N_CLASSES):
                        output_merge[k] *= network_results[z][i][k]
                output.append((label_list[i], LanguageGroup.LABEL_MAP_STR[max_index(output_merge)+1]))

            return output



class _BNN (ClassifierBase):
    def __init__(self,ngram_range=(1,4),binary=True,analyzer='word',lower=True,feature_count=FEATURE_COUNT):
        super(_BNN, self).__init__()
        self.estimator = None
        self.kbest = SelectKBest(k=feature_count)
        self.cv = CountVectorizer(ngram_range=ngram_range,binary=binary,lowercase=lower,analyzer=analyzer)
        self.sel = SelectFromModel(LinearSVC(C=0.47))
        self.tfid = TfidfTransformer()
        self.pipe_list = []

    def preprocess(self, data):
        return data

    def add_pipe(self, pipe):
        self.pipe_list.append(pipe)

    def train(self, training_data, feature_list=None):
        from keras.models import Sequential
        from keras.layers import Dense, Activation, Dropout
        from keras.utils import to_categorical
        from keras.callbacks import EarlyStopping

        feature_counts = []
        label_list = []
        if feature_list is not None:
            # format data for DNN
            for data in feature_list[0]:
                label_list.append(LanguageGroup.LABEL_MAP[data[0][2]] - 1)

            feature_mtx = []
            i = 0
            for feature in feature_list:
                data_list = []
                for data in feature:
                    data_list.append(data[1])

                if i < len(self.pipe_list):
                    feature_mtx.append(self.pipe_list[i].fit_transform(data_list,label_list).toarray())
                i += 1

            feature_counts = feature_mtx[0]
            for i in range(1,len(feature_mtx)):
                feature_counts = np.concatenate([feature_counts,feature_mtx[i]],axis=1)

        else:
            # format data for DNN
            data_list = []
            for data in training_data:
                label_list.append(LanguageGroup.LABEL_MAP[data[0][2]]- 1)
                data_list.append(data[1])

            # extract features
            feature_counts = self.tfid.fit_transform(self.kbest.fit_transform(self.cv.fit_transform(data_list), label_list),label_list).toarray()

        print("feature count: ", feature_counts.shape[1])
        self.estimator = Sequential()
        self.estimator.add(Dense(128,activation='tanh',input_shape=(feature_counts.shape[1],)))
        self.estimator.add(Dropout(0.2))
        self.estimator.add(Dense(N_CLASSES,activation="softmax"))
        self.estimator.compile(loss="categorical_crossentropy",optimizer="Adam", metrics=['accuracy'])

        self.estimator.fit(feature_counts,to_categorical(label_list,num_classes=11),
                           batch_size=128,epochs=100,callbacks=[EarlyStopping(monitor="loss",min_delta=0.1)])

    def classify(self, testing_data, feature_list=None):
        output = []
        feature_counts = None
        if feature_list is not None:
            if feature_list is not None:

                feature_mtx = []
                i = 0
                for feature in feature_list:
                    data_list = []
                    for data in feature:
                        data_list.append(data[1])

                    if i < len(self.pipe_list):
                        feature_mtx.append(self.pipe_list[i].transform(data_list).toarray())
                    i += 1

                feature_counts = feature_mtx[0]
                for i in range(1, len(feature_mtx)):
                    feature_counts = np.concatenate([feature_counts, feature_mtx[i]], axis=1)
        else:
            data_list = []
            label_list = []
            for test in testing_data:
                label_list.append(test[0])
                data_list.append(test[1])

            feature_counts = self.tfid.transform(self.kbest.transform(self.cv.transform(data_list))).toarray()

        results = self.estimator.predict(feature_counts)
        return results
