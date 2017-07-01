from .ClassifierBase import ClassifierBase, LanguageGroup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, SelectFromModel
from src.util.Misc import max_index, select_feature
import numpy as np
import tensorflow as tf

FEATURE_COUNT = "all"
FEATURE = "lemmas"
N_CLASSES = 11


class BNN:
    def __init__(self):
        super(BNN,self).__init__()
        self.network_list = [_BNN(ngram_range=(1,1),lower=False,binary=False)]#_BNN(binary=True,analyzer="char_wb",feature_count='all'), _BNN(binary=True),_BNN(binary=False),_BNN(binary=True, analyzer='char'),
                             #_BNN(binary=False, analyzer='char')]
    def preprocess(self,data):
        return data

    def train(self,training_data):
        if len(training_data) == 1:
            training_data = select_feature(training_data[0][1],training_data[0][0],feature_name=FEATURE)

        i = 0
        for network in self.network_list:
            print("Training Network: ", i+1, " of: ", len(self.network_list))
            network.train(training_data)
            i += 1

    def classify(self,testing_data):
        if len(testing_data) == 1:
            testing_data = select_feature(testing_data[0][1],testing_data[0][0],feature_name=FEATURE)

        label_list = []
        for test in testing_data:
            label_list.append(test[0])

        network_results = []
        for network in self.network_list:
            network_results.append(network.classify(testing_data))

        output = []
        for i in range(0, len(network_results[0])):
            output_merge = network_results[0][i]
            for z in range(1, len(network_results)):
                for k in range(0, N_CLASSES):
                    output_merge[k] *= network_results[z][i][k]
            output.append((label_list[i], LanguageGroup.LABEL_MAP_STR[max_index(output_merge)+1]))

        for out in output:
            print(out[0],out[1])

        return output



class _BNN (ClassifierBase):
    def __init__(self,ngram_range=(1,4),binary=True,analyzer='word',lower=True,feature_count=FEATURE_COUNT):
        super(_BNN, self).__init__()
        self.estimator = None
        self.kbest = SelectKBest(k=feature_count)
        self.cv = CountVectorizer(ngram_range=ngram_range,binary=binary,lowercase=lower,analyzer=analyzer)
        self.sel = SelectFromModel(LinearSVC(C=0.47))

    def preprocess(self, data):
        return data

    def train(self, training_data):
        from keras.models import Sequential
        from keras.layers import Dense, Activation, Dropout
        from keras.utils import to_categorical
        from keras.callbacks import EarlyStopping

        # format data for DNN
        data_list = []
        label_list = []
        for data in training_data:
            label_list.append(LanguageGroup.LABEL_MAP[data[0][2]]- 1)
            data_list.append(data[1])

        # extract features
        feature_counts = self.kbest.fit_transform(self.cv.fit_transform(data_list), label_list).toarray()

        print("feature count: ", feature_counts.shape[1])
        self.estimator = Sequential()
        self.estimator.add(Dense(128,activation='tanh',input_shape=(feature_counts.shape[1],)))
        self.estimator.add(Dropout(0.2))
        self.estimator.add(Dense(N_CLASSES,activation="softmax"))
        self.estimator.compile(loss="categorical_crossentropy",optimizer="Adam", metrics=['accuracy'])

        self.estimator.fit(feature_counts,to_categorical(label_list,num_classes=11),
                           batch_size=128,epochs=100,callbacks=[EarlyStopping(monitor="loss",min_delta=0.001)])

    def classify(self, testing_data):
        output = []
        data_list = []
        label_list = []
        for test in testing_data:
            label_list.append(test[0])
            data_list.append(test[1])

        feature_counts = self.kbest.transform(self.cv.transform(data_list)).toarray()

        results = self.estimator.predict(feature_counts)
        return results
