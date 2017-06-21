import pandas as pd

import os
import re

def df_concat(dir):
    if os.path.isdir(dir):
        final_df = pd.DataFrame()
        for file in os.listdir(dir):
            if re.match('.*.csv', file) is not None:
                print(os.path.basename(file))
                print(os.path.join(dir, file))
                cur_df = pd.read_csv(os.path.join(dir, file))
                final_df = pd.concat([final_df, cur_df])
        return final_df
    else:
        print('Not a directory')

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


if __name__ == '__main__':

    from sys import  argv
    myargs = getopts(argv)
    dir  = myargs['-d']

    df = df_concat(dir)
    df.to_csv(os.path.join(dir, 'train_corrected_cleaned.csv'))