from .ClassifierBase import ClassifierBase, LanguageGroup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
import numpy as np
import tensorflow as tf

FEATURE_COUNT = 12000


class DNNC (ClassifierBase):
    def __init__(self):
        super(DNNC, self).__init__()
        try:
            import tensorflow as tf
            self.tf = tf
            tf.logging.set_verbosity(tf.logging.INFO)
        except ImportError as e:
            print("Could not import tensor flow", str(e))
        self.estimator = None
        self.kbest = SelectKBest(k=FEATURE_COUNT)
        self.cv = CountVectorizer(ngram_range=(1,2))

    def preprocess(self, data):
        return data

    def train(self, training_data):
        # format data for DNN
        data_list = []
        label_list = []
        for data in training_data:
            label_list.append(str(LanguageGroup.LABEL_MAP[data[0][2]]))
            data_list.append(data[1])

        # extract features
        feature_counts = self.kbest.fit_transform(self.cv.fit_transform(data_list), label_list).toarray()

        feature_columns = [self.tf.contrib.layers.real_valued_column("x", dimension=1)]

        self.estimator = self.tf.contrib.learn.DNNClassifier([12000],
                                       feature_columns=feature_columns,
                                       n_classes=11,
                                       activation_fn=tf.nn.tanh,
                                       optimizer=tf.train.AdamOptimizer(),
                                       label_keys=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
                                       model_dir="/tmp/model/", dropout=0.2)

        def input_func():
            x_data = feature_counts
            y_data = np.array(label_list)
            x = tf.constant(x_data, shape=x_data.shape, verify_shape=True, dtype=tf.float32)
            y = tf.constant(y_data, shape=y_data.shape, verify_shape=True, dtype=tf.string)

            x, y = tf.train.slice_input_producer([x, y], num_epochs=6)
            slice_xy = dict(x=x, y=y)
            batch_xy = tf.train.batch(slice_xy, batch_size=100, capacity=200, allow_smaller_final_batch=True)
            batch_label = batch_xy.pop('y')
            return batch_xy, batch_label

        self.estimator.fit(input_fn=input_func)

    def classify(self, testing_data):
        output = []
        data_list = []
        label_list = []
        for test in testing_data:
            label_list.append(test[0])
            data_list.append(test[1])

        feature_counts = self.kbest.transform(self.cv.transform(data_list)).toarray()

        def test_func():
            x = tf.constant(np.array(feature_counts), dtype=tf.float32)
            return {"x": x}

        i = 0
        result = list(self.estimator.predict_classes(input_fn=test_func))
        for res in result:
            output.append((label_list[i], LanguageGroup.LABEL_MAP_STR[int(res)]))
            i += 1

        print(str(output))
        return output
