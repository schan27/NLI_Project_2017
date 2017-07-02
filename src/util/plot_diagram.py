import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import re


lang = ['ARA',  'CHI',  'FRE',  'GER',  'HIN',  'ITA',  'JPN',  'KOR',  'SPA', 'TEL',  'TUR']
def count_token_per_sent(full):
    freq = []
    charfreq =[]
    #[[] for i in range(11)]
    # print(len(freq), len(freq[3]))
    for i in range(11):
        # l = np.zeros((1, 11))
        # l[0,i] = 1
        # l.tostring()
        rows = full.loc[full['label'] == i]
        # print(rows[:10])
        texts = rows.sent.tolist()
        lenfreq = [len(t.split()) for t in texts]
        charfreq.append([len(t) for t in texts])
        freq.append(lenfreq)
    print(len(freq), len(freq[3]), len(freq[4]), freq[3], '\n', freq[4])
    return freq, charfreq


def count_token_per_doc(df):
    freq = []
    for i in range(11):
        rows = df.loc[df['L1'] == lang[i] ]
        texts = rows.corrected.tolist()
        # lenfreq = [len(t.split())]
        lenfreq = [len(re.findall(r'\w+', t)) for t in texts]
        freq.append(lenfreq)

    return freq


def plot_bar_graph(a_mean, a_std, b_mean, b_std):
    N = len(a_mean)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, a_mean, width, color='r', yerr=a_std)
    rects2 = ax.bar(ind + width, b_mean, width, color='y', yerr=b_std)

    ax.set_ylabel('# tokens')
    ax.set_title('Number of Tokens per document or sentence')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('ARA',  'CHI',  'FRE',  'GER',  'HIN',  'ITA',  'JPN',  'KOR',  'SPA', 'TEL',  'TUR'))

    ax.legend((rects1[0], rects2[0]), ('doc.', 'sent.'))
    # ax.legend.FontSize(12)
    # set(ax.legend, 'FontSize', 12)
    
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            # ax.text(rect.get_x() + rect.get_width() / 2., 1.22 * height,
            #         '%d' % int(height),
            #         ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.show()


if __name__ == '__main__':


    train = pd.read_csv('../../nli-shared-task-2017/data/essays/' \
                        'train/lstm_in/train_corrected_cleaned.csv')
    dev = pd.read_csv('../../nli-shared-task-2017/data/essays/' \
                        'dev/lstm_in/dev_corrected_cleaned.csv')

    labels = train.label.tolist()
    labels = [np.argmax(np.array(ast.literal_eval(li))) for li in labels]
    train.label = labels

    labels = dev.label.tolist()
    labels = [np.argmax(np.array(ast.literal_eval(li))) for li in labels]
    dev.label = labels

    print(train[:5])


    full = pd.concat([train, dev])

    freq, charfreq = count_token_per_sent(full)
    mean_sent = [np.mean(x) for x in freq]
    std_sent = [np.std(x) for x in freq]

    mean_char = [np.mean(x) for x in charfreq]
    std_char = [np.std(x) for x in charfreq]
    print('char', mean_char)
    print(std_char)

    print(mean_sent)
    print(['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR'])

    #################
    train = pd.read_csv('../../nli-shared-task-2017/data/essays/' \
                        'train/corrected_csv/train_corrected.csv')
    labels = pd.read_csv('../../nli-shared-task-2017/data/labels/train/labels.train.csv')

    data = pd.merge(train, labels, how='inner', left_on='id', right_on='test_taker_id')


    freq = count_token_per_doc(data)
    print(len(freq[0]), len(freq[0]), len(freq[0]))
    mean_doc = [np.mean(x) for x in freq]
    std_doc = [np.std(x) for x in freq]
    print(mean_doc)

    # plot_bar_graph(mean_doc, std_doc, mean_sent, std_sent)





