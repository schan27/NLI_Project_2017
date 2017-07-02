import pandas as pd
import numpy as np
import os
import re

lang = {'ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR'}
langdict = {'ARA': 0, 'CHI': 1, 'FRE': 2, 'GER': 3, 'HIN': 4, 'ITA': 5,
            'JPN': 6, 'KOR': 7, 'SPA': 8, 'TEL': 9, 'TUR': 10}


def raw2csv(dirname, labeldir):
    if os.path.isdir(dirname):

        essays = []
        ids = []
        labels = []
        df = pd.read_csv(labeldir)

        for file in os.listdir(dirname):
            if re.match(r".*\.txt$", file) is not None:
                with open(os.path.join(dirname, file), 'r') as f:
                    e = str(f.read())
                    essays.append(e.replace(';', ''))
                    fnum = int(os.path.splitext(file)[0])
                    ids.append(fnum)

                    cur_row = df.loc[df['test_taker_id'] == fnum]
                    l = cur_row.L1.tolist()[0]
                    labels.append(langdict[l])
                    #                     print(file)
                    #                     print(ids)
                    #                     print(labels)

                    #                     break
        return ids, essays, labels
    else:
        print('path must be a directory')
        return None, None, None




track = 'dev'
ids, esseys, labels = raw2csv('../../nli-shared-task-2017/data/essays/' + track + '/tokenized',
                      '../../nli-shared-task-2017/data/labels/' + track + '/labels.' + track + '.csv')
if ids == None or esseys == None or labels == None:
    print('one of the arrays were None')
    exit()

print(len(ids), ids[:10])
print(len(esseys), esseys[:10])
print(len(labels), labels[:10])

data = pd.DataFrame({'test_taker_id': ids, 'essay': esseys, 'label': labels} , index = None)
data.to_csv('../../nli-shared-task-2017/data/essays/' + track + '/raw_csv/' + track + '.csv', index=None)
