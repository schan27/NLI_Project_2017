from src.Classifiers.ClassifierBase import ClassifierBase
from src.Classifiers.BNNClassifier import BNN

class Ensemble(ClassifierBase):
    def __init__(self):
        super(Ensemble,self).__init__()
        self.bnn = BNN(True)


    # do pre processing here. (may be called multiple times if, multiple inputs)
    # return the processed data.
    def preprocess(self,data):
        raise NotImplementedError()

    # train on the given list of tuples EX: [(label,data), (label,data)]
    # where label = (speech_prompt, essay_prompt, L1) and data is document text.
    # return None.
    def train(self,training_data):
        self.bnn.train(training_data)

    # test on the given list of tuples EX: [(label,data), (label,data)]
    # where label = (speech_prompt, essay_prompt, L1) and data is document text.
    # return list of [label ,LanguageGroup.XXX].
    # return the (predicted) language group of the input
    def classify(self,testing_data):
        self.bnn.classify(testing_data)
