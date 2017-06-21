"""purpose of file:
save a csv file that on each row there is a
cleaned sentence and it's label(in number)

"""
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.utils.np_utils import to_categorical
import os, re
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random
import sys

# local imports
# from Spelling_correction import correct_spelling
from Spelling_correction import *

NB_CLASSES = 11

lang = {'ARA',  'CHI',  'FRE',  'GER',  'HIN',  'ITA',  'JPN',  'KOR',  'SPA', 'TEL',  'TUR'}
langdict = {'ARA': 0,  'CHI': 1,  'FRE': 2,  'GER': 3,  'HIN': 4,  'ITA': 5,
         'JPN': 6,  'KOR': 7,  'SPA': 8,  'TEL': 9,  'TUR': 10}

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

def clean_str(text, to_lower =False, remove_stopwords=False, stem_words=False):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    if to_lower:
        text = text.lower().split()
    else:
        text = text.split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)
    string = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    string = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)


    return text



def clean_tokenize(target):
    if os.path.isfile(target):
        print('Tokenizing {} ...'.format(target))# ---
        return __clean_tokenize()
    elif os.path.isdir(target):
        for file in os.listdir(target):
            if re.match(r".*\.c$", file) is not None:
                if not __clean_tokenize(os.path.join(target, file)):
                    return False
        print('finished cleaning files.')
        return True
    else:
        raise FileNotFoundError



def __clean_tokenize(target):
    try:
        file_in = open(target, 'r')
        file_out = None
        output_buffer = []
        for line in file_in.readlines():
            output_buffer.append(clean_str(line))

        dir_path = os.path.dirname(target)
        base_name = os.path.basename(target)
        print('basename:', base_name) #---

        new_dir = dir_path + '/tokenized_clean/'

        # create the path if it doesn't exist
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        file_out = open(new_dir+base_name, 'w+')
        file_out.writelines(output_buffer)

    except IOError as e:
        print("ERROR: when reading file: " + target + " error message: " + str(e))
        return False

    finally:
        if file_in is not None:
            file_in.close()
        if file_out is not None:
            file_out.close()
        return True

def load_data(to_cat = True, verbose = True, classes = lang, correct = True,
              labelpath = '../../nli-shared-task-2017/data/' \
                          'labels/train/labels.train.csv',
              datapath = '../../nli-shared-task-2017/data/essays/train'\
                        '/tokenized',
              outpath = './'
              ):
    x = np.array([])
    y = np.array([])
    df = pd.read_csv(labelpath)
    columns = df.columns.tolist()
    columns = ['sent'] + columns
    idx = 1
    df_out = pd.DataFrame(index = None, columns=columns)
    if os.path.isdir(datapath):
        i =0
        for file in os.listdir(datapath):
            print(file)
            # if re.match(r".*\.txt$", file) is not None:
            if re.match(r".*\.corrected$", file) is not None:
                i+=1
                file_in = None
                if correct:
                    correct_spelling(os.path.join(datapath, file))

                    file_in = open(os.path.join(datapath,'corrected_out', file+ '.corrected'), 'r')
                else:
                    file_in = open(os.path.join(datapath, file), 'r')


                fname = os.path.splitext(os.path.basename(file))[0]
                ###
                fname = os.path.splitext(os.path.basename(fname))[0]
                fnum = int(fname)

                cur_row = df.loc[df['test_taker_id'] == fnum]
                label = cur_row.L1.tolist()[0]
                #read lines in the file
                for line in file_in.readlines():
                    if line.isspace(): continue
                    x = np.append(x, [clean_str(line)])
                    y =np.append(y, [langdict[label]])


            if i % 250 == 0 :
                print(x.shape, y.shape)
                if to_cat:
                    y = to_categorical(y, num_classes=NB_CLASSES)
                save_data(x, y, fname='run_'+str(idx), folder=outpath)
                idx+=1
                x = np.array([])
                y = np.array([])

            # if idx> 1000:
            #     break

                    #can add later
                        #make current row
                        #append to dataframe

        print(x.shape, y.shape)
        if to_cat:
            y = to_categorical(y, num_classes=NB_CLASSES)






    else :
        print('datapath should be a folder!')
        sys.exit(1)


    return x, y


def save_data (x, y, fname='train.csv', folder ='../../nli-shared-task-2017/data/essays/train' \
                        '/sentences/'):
    if not os.path.exists(folder):
        os.mkdir(folder)
    d = {'sent': x.tolist(), 'label': y.tolist()}
    df = pd.DataFrame(data=d, index=None)
    df.to_csv(os.path.join(folder, fname+'.csv'))

if __name__ == '__main__':

    from sys import  argv
    myargs = getopts(argv)
    lpath = myargs['-l']
    dpath = myargs['-d']
    outpath = myargs['-o']
    partition  = myargs['-p']

    x_train, y_train = load_data(labelpath=lpath, datapath= dpath, correct=False, outpath=outpath)

    print(x_train[:10], y_train[:10])
    save_data(x_train, y_train, outpath, fname=partition)
