from src.Classifiers.ClassifierBase import ClassifierBase, LanguageGroup
from src.Classifiers.BNNClassifier import BNN
from src.Classifiers.SVM  import SVM
from src.util.Misc import max_index, select_feature, max_frequency
import numpy as np

class Ensemble(ClassifierBase):
    def __init__(self):
        super(Ensemble,self).__init__()
        self.bnn = BNN(True)
        self.svm = SVM()


    # do pre processing here. (may be called multiple times if, multiple inputs)
    # return the processed data.
    def preprocess(self,data):
        return data

    # train on the given list of tuples EX: [(label,data), (label,data)]
    # where label = (speech_prompt, essay_prompt, L1) and data is document text.
    # return None.
    def train(self,training_data):
        self.bnn.train(training_data)
        self.svm.train(training_data)

    # test on the given list of tuples EX: [(label,data), (label,data)]
    # where label = (speech_prompt, essay_prompt, L1) and data is document text.
    # return list of [label ,LanguageGroup.XXX].
    # return the (predicted) language group of the input
    def classify(self,testing_data):
        output = []
        feature_data = select_feature(testing_data[0][1], testing_data[0][0], "original")
        label_list = []
        for data in feature_data:
            label_list.append(data[0])

        output.extend(self.bnn.classify(testing_data))
        output.extend(self.svm.classify(testing_data))

        #final = np.ndarray(shape=(output[0].shape[0], 4), dtype=object)
        #for j in range(0, len(output)):
        #    for i in range(0, len(output[j])):
        #        final[:, j][i] = list(output[j][i])

        #np.save("foobar_dev", final)

        final_mtx = output[0]
        for i in range(1, len(output)):
            final_mtx = np.add(final_mtx,output[i])

        #final_mtx = np.ndarray((output[0].shape[0],output[0].shape[1]))
        #for i in range(0, len(output[0])):
        #    row_median = [[],[],[],[],[],[],[],[],[],[],[]]
        #    for mtx in output:
        #        for z in range(0, len(mtx[i])):
        #            row_median[z].append(mtx[i][z])

        #    row_median_result = []
        #    for row_l in row_median:
        #        row_median_result.append(np.median(row_l))
        #    final_mtx[i] = row_median_result

        #final_mtx = np.ndarray((output[0].shape[0], len(output)))
        #for i in range(0,len(output[0])):
        #    votes = []
        #    for mtx in output:
        #        votes.append(max_index(mtx[i]))
        #    final_mtx[i] = votes

        result = []
        for row in final_mtx:
            result.append(max_index(row))



        i = 0
        output_final = []
        for res in result:
            output_final.append((label_list[i], LanguageGroup.LABEL_MAP_STR[res + 1]))
            print(str(label_list[i][3])+","+LanguageGroup.LABEL_MAP_STR[res + 1])
            i += 1
        return output_final

