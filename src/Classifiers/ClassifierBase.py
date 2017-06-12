class LanguageGroup:
    ARA = "ARA"
    CHI = "CHI"
    FRE = "FRE"
    GER = "GER"
    HIN = "HIN"
    ITA = "ITA"
    JPN = "JPN"
    KOR = "KOR"
    SPA = "SPA"
    TEL = "TEL"
    TUR = "TUR"
    LABEL_MAP = {"ARA":1,
                 "CHI":2,
                 "FRE":3,
                 "GER":4,
                 "HIN":5,
                 "ITA":6,
                 "JPN":7,
                 "KOR":8,
                 "SPA":9,
                 "TEL":10,
                 "TUR":11}
    LABEL_MAP_STR= {1:"ARA",
                2:"CHI",
                3:"FRE",
                4:"GER",
                5:"HIN",
                6:"ITA",
                7:"JPN",
                8:"KOR",
                9:"SPA",
                10:"TEL",
                11:"TUR"}


class ClassifierBase:
    def __init__(self):
        pass

    # do pre processing here. (may be called multiple times if, multiple inputs)
    # return the processed data.
    def preprocess(self,data):
        raise NotImplementedError()

    # train on the given list of tuples EX: [(label,data), (label,data)]
    # where label = (speech_prompt, essay_prompt, L1) and data is document text.
    # return None.
    def train(self,training_data):
        raise NotImplementedError()

    # test on the given list of tuples EX: [(label,data), (label,data)]
    # where label = (speech_prompt, essay_prompt, L1) and data is document text.
    # return list of [label ,LanguageGroup.XXX].
    # return the (predicted) language group of the input
    def classify(self,testing_data):
        raise NotImplementedError()
