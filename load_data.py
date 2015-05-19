import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.externals.joblib import Memory
memory = Memory('data/cache/')


# @memory.cache
def cat_mapping():
    path = 'data/full/trainingData.csv'
    from collections import defaultdict
    train_sets = defaultdict(set)
    with open(path) as f:
        for line in f:
            splited = line.split(',')
            for click in splited[-1].split(';'):
                click_split = click.split('/')[:-1]
                train_sets['a'].add(click_split[0])
                train_sets['b'].add(click_split[1])
                train_sets['c'].add(click_split[2])
                train_sets['d'].add(click_split[3])

                train_sets['ab'].add(click_split[0] + click_split[1])
                train_sets['abc'].add(click_split[0] + click_split[1]
                                      + click_split[2])
                train_sets['bc'].add(click_split[1] + click_split[2])

    path = 'data/full/testData.csv'
    from collections import defaultdict
    test_sets = defaultdict(set)
    with open(path) as f:
        for line in f:
            splited = line.split(',')
            for click in splited[-1].split(';'):
                click_split = click.split('/')[:-1]
                test_sets['a'].add(click_split[0])
                test_sets['b'].add(click_split[1])
                test_sets['c'].add(click_split[2])
                test_sets['d'].add(click_split[3])

                test_sets['ab'].add(click_split[0] + click_split[1])
                test_sets['abc'].add(click_split[0] + click_split[1]
                                     + click_split[2])
                test_sets['bc'].add(click_split[1] + click_split[2])

    for set_ in ['a', 'b', 'c', 'd', 'ab', 'abc', 'bc']:
        train_sets[set_] = train_sets[set_].intersection(test_sets[set_])

    cat_to_nr = {}
    for cat in ['a', 'b', 'c', 'd']:
        cat_to_nr[cat] = dict(zip(sorted(train_sets[cat]),
                                  np.arange(len(train_sets[cat]))))
        cat_to_nr[cat]['unknown'] = len(cat_to_nr[cat])

    for cat in ['ab', 'abc', 'bc']:
        cat_to_nr[cat] = dict(zip(sorted(train_sets[cat]),
                                  np.arange(len(train_sets[cat]))))
        cat_to_nr[cat]['unknown'] = len(cat_to_nr[cat])

    nr_to_cat = {}
    for cat in ['a', 'b', 'c', 'd']:
        nr_to_cat[cat] = {value: key for key, value in cat_to_nr[cat].items()}
    for cat in ['ab', 'abc', 'bc']:
        nr_to_cat[cat] = {value: key for key, value in cat_to_nr[cat].items()}
    return cat_to_nr, nr_to_cat


@memory.cache
def freq_mapping(min_freq=5):
    path = 'data/full/trainingData.csv'
    records = []

    with open(path) as f:
        for nr, line in enumerate(f):
            splited = line.split(',')
            session_id = splited[0]
            for click in splited[-1].split(';'):
                levels = click.split('/')
                records.append((session_id, levels[0], levels[1],
                                levels[2], levels[3]))
    df = DataFrame.from_records(records, columns=['session_id',
                                                  'a', 'b', 'c', 'd'])
    cat_to_nr = {}
    nr_to_cat = {}
    df['dummy'] = 1
    for cat in ['a', 'b', 'c', 'd']:
        df_tmp = df[['dummy', cat]].groupby(cat).count()
        ids = df_tmp[df_tmp.dummy > min_freq].index.values.astype(str)
        cat_to_nr[cat] = dict(zip(sorted(ids), np.arange(len(ids))))
        cat_to_nr[cat]['unknown'] = len(cat_to_nr[cat])
        nr_to_cat[cat] = {value: key for key, value in cat_to_nr[cat].items()}
    return cat_to_nr, nr_to_cat


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def data_df(path='data/', is_test=False):
    if is_test:
        train_data = pd.read_csv(path + 'testData.csv', header=None,
                                 parse_dates=[1, 2],
                                 names=['session_id', 'start', 'end', 'cats'])
    else:
        train_data = pd.read_csv(path + 'trainingData.csv', header=None,
                                 parse_dates=[1, 2],
                                 names=['session_id', 'start', 'end', 'cats'])
        del train_data['cats']
        train_label = pd.read_csv(path + 'trainingLabels.csv', header=None)
        train_data['gender'] = (train_label[0].values
                                == 'female').astype(np.float64)
    return train_data
