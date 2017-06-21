from nltk.tag import StanfordPOSTagger
from nltk.tag.perceptron import PerceptronTagger
import nltk.data, nltk.tag

from nltk.parse import stanford
from os import path, listdir, mkdir
import os
import re
import pickle

# targets single file or a directory
# if the target is a directory all .txt files in the directory
# will have a new file in pos directory containing the taggs

#dependency parsing is not working yet
#if is_sent = True it does dedependency parsind

#if you whant to run dependency tree locally,
# please download  stanford-parser-3.7.0-models.jar

jar = '../lib/Stanford_POS/stanford-postagger.jar'
model = '../lib/Stanford_POS/english-bidirectional-distsim.tagger'
os.environ['STANFORD_PARSER'] = '../lib/Stanford_POS/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '../lib/Stanford_POS/stanford-parser-3.7.0-models.jar'
# st = StanfordPOSTagger(model, jar, encoding='utf8')
dep = stanford.StanfordParser(model_path='../lib/Stanford_POS/englishPCFG.ser.gz')
st = PerceptronTagger()



def tag_pos(target, is_sent = False):
    if path.isfile(target):
        return __tag_pos(target, is_sent)
    elif path.isdir(target):
        for file in listdir(target):
            if re.match(r".*\.txt$", file) is not None:
                if not __tag_pos(path.join(target, file), is_sent):
                    return False
        return True
    else:
        raise FileNotFoundError




def __tag_pos(target, is_sent):
    file_in = None
    file_out = None

    try:
        file_in = open(target, 'r')
        output_buffer = []
        for line in file_in.readlines():
            if is_sent:
                output_buffer.append(dep.raw_parse(line))
            else:
                output_buffer.append(st.tag(line.split()))
            # print(line)



        # file_path = path.splitext(target)
        dir_path = path.dirname(target)
        base_name = path.splitext(path.basename(target))[0]

        new_dir = dir_path+'/POS/'  
        if not path.exists(new_dir):
            mkdir(new_dir)

        file_out = open(new_dir + base_name + '_pos.pkl', 'wb')
        pickle.dump(output_buffer, open(new_dir + base_name + '_pos.pkl', 'wb'), protocol=2)
        print(new_dir + base_name + '_pos.pkl')

    except IOError as e:
        print("ERROR: when reading file: " + target + " error message: " + str(e))
        return False
    finally:
        if file_in is not None:
            file_in.close()
        if file_out is not None:
            file_out.close()
    return True

tag_pos('../../nli-shared-task-2017/data/essays/dev/tokenized')
