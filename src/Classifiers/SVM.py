from sklearn.svm import LinearSVC
from src.util.Misc import select_feature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np


def normalize(x, maximum, minimum):
    return (x - minimum) / (maximum - minimum)


class SVM():
    
    def __init__(self):
        self.pipeline = Pipeline(
            [('vectorizer', TfidfVectorizer(binary=True, norm='l2')),
            ('classifier', LinearSVC())])
    
    def preprocess(self, data):
        return data
    
    def train(self, training_data):
        result = select_feature(training_data[0][1],training_data[0][0],"original")
        info, data = zip(*result)
        labels = [item[2] for item in info]
        self.pipeline.fit(data, labels)
        
    def classify(self, testing_data):
        result = select_feature(testing_data[0][1], testing_data[0][0], "original")
        info, data = zip(*result)
        predictions = self.pipeline.decision_function(data)
        minimum = np.min(predictions)
        maximum = np.max(predictions)
        
        for i, arr in enumerate(predictions):
            for j, cell in enumerate(arr):
                predictions[i][j] = normalize(cell, maximum, minimum)
        
        return predictions
        