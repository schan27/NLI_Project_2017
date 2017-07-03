from sklearn.svm import LinearSVC
from src.util.Misc import select_feature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np


def normalize(x, maximum, minimum):
    return (x - minimum) / (maximum - minimum)


class SVM():
    
    def __init__(self):
        self.clf_list = [
            _SVM('original', 'word', 2), 
            _SVM('original', 'word', 3),
            _SVM('original', 'char', 4), 
            _SVM('original', 'char', 5),
            _SVM('phonemes', 'word', 4), 
            _SVM('phonemes', 'word', 5),
            ]
            
    def preprocess(self, data):
        return data
        
    def train(self, training_data):
        for clf in self.clf_list:
            clf.train(training_data)
    
    def classify(self, testing_data):
        result = [clf.classify(testing_data) for clf in self.clf_list]
        return result
        
        
class _SVM():
    
    def __init__(self, col_name, analyzer, n):
        tokenize = lambda x: x.split()
    
        self.pipeline = Pipeline(
            [('vectorizer', TfidfVectorizer(tokenizer=tokenize, binary=True, norm='l2', ngram_range=(n, n), analyzer=analyzer)),
            ('classifier', LinearSVC())])
            
        self.col_name = col_name
    
    def preprocess(self, data):
        return data
    
    def train(self, training_data):
        result = select_feature(training_data[0][1],training_data[0][0], self.col_name)
        info, data = zip(*result)
        labels = [item[2] for item in info]
        self.pipeline.fit(data, labels)
        
    def classify(self, testing_data):
        result = select_feature(testing_data[0][1], testing_data[0][0], self.col_name)
        info, data = zip(*result)
        predictions = self.pipeline.decision_function(data)
        minimum = np.min(predictions)
        maximum = np.max(predictions)
        
        for i, arr in enumerate(predictions):
            for j, cell in enumerate(arr):
                predictions[i][j] = normalize(cell, maximum, minimum)
        
        return predictions
        