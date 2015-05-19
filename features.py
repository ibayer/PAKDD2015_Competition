import load_data
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.externals.joblib import Memory
memory = Memory('data/cache/')

nr_levels = {'a': 11, 'b': 91, 'c': 439, 'd': 36122}


#@memory.cache
def cat_matrices(path='data/', is_test=False, min_freq=0):
    path += 'testData.csv' if is_test else'trainingData.csv'
    if min_freq > 0:
        cat_to_nr, _ = load_data.freq_mapping(min_freq)
        tmp, _ = load_data.cat_mapping()
        cat_to_nr['ab'] = tmp['ab']
        cat_to_nr['bc'] = tmp['bc']
        cat_to_nr['abc'] = tmp['abc']
    else:
        cat_to_nr, _ = load_data.cat_mapping()
    n_lines = load_data.file_len(path)

    dok_cats = {cat: sp.lil_matrix((n_lines, len(cat_to_nr[cat])),
                dtype=np.float64) for cat in ['a', 'b', 'c', 'd']}
    unique_items = []

    with open(path) as f:
        for nr, line in enumerate(f):
            splited = line.split(',')
            unique_items.append(splited[-1].count(';') + 1)
            for click in splited[-1].split(';'):

                cs = click.split('/')[:-1]
                dok_cats['a'][nr, cat_to_nr['a'][cs[0]]] += 1
                if cs[1] in cat_to_nr['b']:
                    dok_cats['b'][nr, cat_to_nr['b'][cs[1]]] += 1
                else:
                    dok_cats['b'][nr, cat_to_nr['b']['unknown']] += 1
                if cs[2] in cat_to_nr['c']:
                    dok_cats['c'][nr, cat_to_nr['c'][cs[2]]] += 1
                else:
                    dok_cats['c'][nr, cat_to_nr['c']['unknown']] += 1
                if cs[3] in cat_to_nr['d']:
                    dok_cats['d'][nr, cat_to_nr['d'][cs[3]]] += 1
                else:
                    dok_cats['d'][nr, cat_to_nr['d']['unknown']] += 1

    if min_freq > 0:
        for cat in ['a', 'b', 'c', 'd']:
            dok_cats[cat + '_freq' + str(min_freq)] = dok_cats.pop(cat)
    return dok_cats


def get_range(id_, pad):
    max_ = {'A': 11, 'B': 91, 'C': 439, 'D': 33489}
    max_ = {'A': 11, 'B': 91, 'C': 439, 'D': 36122}
    pos = int(id_[1:])
    return range(max(0, pos - pad), min(max_[id_[0]], pos + pad))


def dummy_encoding(x, n_levels):
    """ assumes values are in [0, n_levels]
    """
    n_samples = len(x)
    n_features = n_levels
    X = sp.dok_matrix((n_samples, n_features), dtype=np.float64)
    for i in range(len(x)):
        X[i, x[i]] = 1
    return X.tocsc()


def neighbors(cats, doc, is_test):
    max_shift = doc['max_shift']
    results = {}
    for f in doc['shift_features']:
        X = cats[f].tocsr()
        n_samples, n_features = X.shape
        X_coll = None
        for dist in range(1, max_shift + 1):
            X_left, X_right, X_shift = matrix_shift(X, dist)
            if X_coll is None:
                X_coll = X_shift
            else:
                X_coll = X_coll + X_shift
                results[f + '1-' + str(dist)] = X_shift
            assert X_shift.shape == X.shape
            results[f + str(dist)] = X_shift
    return results


def matrix_shift(X, shift):
    n_samples = X.shape[0]
    i_shift = np.arange(n_samples) + shift
    padding = i_shift > n_samples - 1
    i_shift[padding] = 0
    X_right = X[i_shift, :]
    X_right[padding, :] = 0

    i_shift = np.arange(n_samples) - shift
    padding = i_shift < 0
    i_shift[padding] = 0
    X_left = X[i_shift, :]
    X_left[padding, :] = 0
    return X_left, X_right, X_left + X_right
