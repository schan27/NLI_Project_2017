import os
import re
import pandas
from sklearn.metrics import f1_score

class ClassifierFrameWork:
    def __init__(self):
        self.classifiers = [] # list of classifiers
        self.train_data = [] # (label, data)
        self.test_data = [] # (label, data)
        self.results = []  # (label, L1, str_classifier_type)
        self.label_map = None # map of all labels {file_name:(speech_prompt,essay_prompt,L1)}

    def load_data_from_file(self, path, bTest=False):
        if len(self.label_map) == 0:
            print("ERROR: you must load labels before loading any data files")
            return

        if os.path.splitext(path)[1] == ".csv":
            csv_data = pandas.read_csv(path)
            if bTest:
                self.test_data.append((self.label_map,csv_data))
            else:
                self.train_data.append((self.label_map,csv_data))
        else:
            file = None
            data_buffer = ""
            try:
                file = open(path,"r")
                data_buffer = file.read()
            except IOError as e:
                print("ERROR: " + str(e))
                return
            finally:
                if file is not None:
                    file.close()

            data_label = self.label_map[re.match(r"^[0-9]+",os.path.basename(path)).group(0)]
            if data_label is not None:
                if bTest:
                    self.test_data.append((data_label,data_buffer))
                else:
                    self.train_data.append((data_label,data_buffer))
            else:
                if bTest:
                    self.test_data.append((("Foo","Foo","Foo","Foo"), data_buffer))
                else:
                    self.train_data.append((("Foo", "Foo", "Foo", "Foo"), data_buffer))

                print("ERROR: could not find label for: " + path)

    # load data labels from file
    def load_label_file(self,path):
        if self.label_map is None:
            self.label_map = {}

        file = None
        data_buffer = ""
        try:
            file = open(path, "r")
            data_buffer = file.read()
        except IOError as e:
            print("ERROR: " + str(e))
            return
        finally:
            if file is not None:
                file.close()

        # process label data
        for line in data_buffer.split("\n"):
            if len(line) != 0:
                items = line.split(",")
                self.label_map[items[0]] = (items[1],items[2],items[3],items[0])

    def preprocess_data(self):
        new_train = []
        new_test = []
        for classifier in self.classifiers:
            for data in self.train_data:
                new_train.append((data[0],classifier.preprocess(data[1])))
            for data in self.test_data:
                new_test.append((data[0],classifier.preprocess(data[1])))

        self.train_data = new_train
        self.test_data = new_test

    def train(self):
        # train
        for classifier in self.classifiers:
            classifier.train(self.train_data)

    def test(self):
        self.results = []
        for classifier in self.classifiers:
            self.results.extend(classifier.classify(self.test_data))

    def check_results(self):
        correct = 0
        total = 0
        true_list = []
        pred_list = []
        for result in self.results:
                total += 1
                if result[0][2] == result[1]:
                    correct +=1
                true_list.append(result[0][2])
                pred_list.append(result[1])

        print("F1 Score: ",f1_score(true_list, pred_list,average='weighted'))
        print("Overall Classification Accuracy is: " + str(float(correct)/float(total)))

    def add_classifier(self, new_classifier):
        if new_classifier in self.classifiers:
            raise AttributeError("new classifier already in classifiers list")
        self.classifiers.append(new_classifier)

    def add_training_data(self, data):
        if type(data) != tuple:
            raise ValueError("must be tuple (file path, data)")
        self.train_data.append(data)

    def add_testing_data(self, data):
        if type(data) != tuple:
            raise ValueError("must be tuple (file path, data)")
        self.test_data.append(data)

    def clear_data(self):
        self.train_data = []
        self.test_data = []

    def clear_classifiers(self):
        self.classifiers = []

    def clear_results(self):
        self.results = []

    def clear_label_map(self):
        self.label_map = None

    def clear_all(self):
        self.train_data = []
        self.test_data = []
        self.results = []
        self.classifiers = []
        self.label_map = None
