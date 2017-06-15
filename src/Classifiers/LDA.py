from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, mutual_info_classif
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .ClassifierBase import ClassifierBase, LanguageGroup
from src.util.Misc import max_frequency, split_on_sentence


class LDA(ClassifierBase):
    def __init__(self):
        super(LDA,self).__init__()
        self.pipe = Pipeline([('cv',CountVectorizer(ngram_range=(1,4),binary=False)),
                              ('svc', SelectFromModel(LinearSVC(C=0.47))),
                              ('sel',SelectKBest(k=8000)),
                              ('var', VarianceThreshold()),
                              ('dens',DenseTransformer()),
                              ('lda',LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])

    def preprocess(self,data):
        return data

    def train(self,training_data):
        # format data for lda
        data_list = []
        label_list = []
        for data in training_data:
            sentences = split_on_sentence(data[1])
            for i in range(0,len(sentences)):
                label_list.append(LanguageGroup.LABEL_MAP[data[0][2]])
            data_list.extend(sentences)

        #grid_search = GridSearchCV(self.pipe,{},n_jobs=2)
        #self.pipe = grid_search.fit(data_list,label_list)
        #print(str(grid_search.best_params_))
        self.pipe.fit(data_list,label_list)

    def classify(self,testing_data):
        import re
        output = []
        for test in testing_data:
            sentences = split_on_sentence(test[1])
            res = self.pipe.predict(sentences)
            output.append((test[0],LanguageGroup.LABEL_MAP_STR[max_frequency(res)]))

        return output

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self,deep=False):
        return {}