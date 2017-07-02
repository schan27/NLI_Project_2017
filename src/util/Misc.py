import glob
import re
from .Argument import Argument, ArgumentType

def expand_paths(arg_list):
    expanded_files = []
    for train_file in arg_list:
        if "*" in train_file.get_string():
            glist = glob.glob(train_file.get_string())
            for path in glist:
                expanded_files.append(Argument(path, ArgumentType.RAW_STRING))
        else:
            expanded_files.append(train_file)
    return expanded_files

# return integer of maximum frequency
def max_frequency(int_list):
    count_map ={}
    for i in int_list:
        if i in count_map:
            count_map[i] = count_map[i] + 1
        else:
            count_map[i] = 1

    max = 0
    best_key = 0
    for key in count_map.keys():
        if count_map[key] > max:
            max = count_map[key]
            best_key = key
    return best_key


def split_on_sentence(str_text):
    return list(map(lambda x: x.replace("\n",""), filter(lambda x: True if len(x) > 1 else False, re.findall(r"[^.!?]+[.!?$]*",str_text))))

def max_index(lst):
    max_v = None
    max_x = None
    x = 0
    for i in lst:
        if ( max_v == None and max_x == None ) or (i > max_v):
            max_v = i
            max_x = x
        x += 1
    return max_x

def select_feature(panda_d_frame,label_map, feature_name="original"):
    feature_list = panda_d_frame.get(feature_name).tolist()
    # windows sucks
    label_list = panda_d_frame.get("Unnamed: 0").tolist()
    # look up label
    label_list = list(map(lambda x: label_map["%.5d" % x], label_list))
    res = list(zip(label_list,feature_list))
    res = list(map(lambda x: (x[0],"blank") if type(x[1]) != str else x,res))
    return res