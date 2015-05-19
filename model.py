import load_data
import blocks
import features
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
import copy
from sklearn.externals.joblib import Memory
memory = Memory('data/cache/')


def evaluate(doc):
    train_data = load_data.data_df(doc['dataset']['path'])
    n_session = train_data.session_id.nunique()
    session_ids = train_data.session_id.unique()
    train_data['row_nr'] = train_data.index
    train_data = train_data.set_index('session_id')

    design_doc = dict((d, copy.deepcopy(doc[d]))
                      for d in ['dataset', 'shift_features', 'max_shift',
                                'min_freq', 'features'])
    train_doc = {'fm_param': doc['fm_param'],
                 'seeds': doc['seeds']}
    y = train_data.gender.values.copy()
    y[y == 0] = -1
    X = create_design_matrix(design_doc)

    i_sessions = np.arange(n_session)
    X_train_org, X_test, i_train, i_test = train_test_split(
        i_sessions, i_sessions, test_size=0.33, random_state=23)

    i_train = train_data.loc[session_ids[i_train]].row_nr.values
    i_test = train_data.loc[session_ids[i_test]].row_nr.values

    y_train_org = y[i_train]
    y_test = y[i_test]
    X_train_org = X[i_train, :]
    X_test = X[i_test, :]

    t_best = 0.0
    acc_best = 0.0
    y_pred_proba = fit_predict(X_train_org, y_train_org, X_test, train_doc)

    df_block = blocks.get_blocks(doc)
    df_block = df_block.set_index('session_id').\
        loc[session_ids[i_test]].reset_index('session_id')

    df_block['proba'] = y_pred_proba
    df_post = blocks.postprocess(df_block, doc['min_size'],
                                 doc['lower_proba'], doc['upper_proba'])
    y_pred_proba = df_post.proba.values

    t_best, acc_best = find_treashold(y_test, y_pred_proba)
    doc['output']['threshold'] = t_best
    doc['output']['acc'] = acc_best
    return doc


def find_treashold(y_test, y_pred_proba,
                   levels=[.5, .65, .7, .75, .7802, .8, .82, .83, .84, .85,
                           .86, .87, .88, .9], verbose=True):
    acc_best = 0
    t_best = 0
    for t in levels:
        acc = balanced_acc(y_test, y_pred_proba, t)
        if verbose:
            print t, acc
        if acc > acc_best:
            t_best = t
            acc_best = acc
    return t_best, acc_best


def submission(doc):
    train_data = load_data.data_df(doc['dataset']['path'])
    y_train = train_data.gender.values.copy()
    y_train[y_train == 0] = -1

    X_train = create_design_matrix(doc, is_test=False)
    X_test = create_design_matrix(doc, is_test=True)

    y_pred_proba = fit_predict(X_train, y_train, X_test, doc)

    df_block = blocks.get_blocks(doc, is_test=True)
    df_block['proba'] = y_pred_proba
    df_post = blocks.postprocess(df_block, doc['min_size'],
                                 doc['lower_proba'], doc['upper_proba'])
    y_pred_proba = df_post.proba.values

    y_output = np.array(['female'] * len(y_pred_proba))
    y_output[y_pred_proba <= doc['threshold']] = 'male'
    np.savetxt(doc['submission_path'], y_output, fmt="%s", newline='\n')
    np.savetxt(doc['submission_path'] + 'probs',
               y_pred_proba, fmt="%s", newline='\n')


def create_design_matrix(doc, is_test=False):
    path = doc['dataset']['path']
    X_cat = features.cat_matrices(path, is_test=is_test)
    if 'min_freq' in doc:
        for min_freq in doc['min_freq']:
            X_cat_min_feq = features.cat_matrices(path, is_test=is_test,
                                                  min_freq=min_freq)
            X_cat.update(X_cat_min_feq)
    X_shift = features.neighbors(X_cat, doc, is_test)
    X_cat.update(X_shift)

    X_features = [normalize(X_cat[cat], norm='l2').tocsc() if
                  sp.isspmatrix(X_cat[cat])
                  else X_cat[cat] for cat in doc['features'] if
                  cat in X_cat]
    X = sp.hstack(X_features).tocsc()
    return X


#@memory.cache
def fit_predict(X_train, y_train, X_test, doc):
    return fit_predict_fm(X_train, y_train, X_test, doc)


def fit_predict_fm(X_train, y_train, X_test, doc):
    print 'fm'
    n_seeds = len(doc['seeds'])
    y_pred_proba = np.empty((X_test.shape[0], n_seeds), dtype=np.float64)

    # fit fm with mutliple seeds, to reduce dependency on individual seeds
    for n, s in enumerate(doc['seeds']):
        from fastFM import mcmc
        param = doc['fm_param']
        fm = mcmc.FMClassification(random_state=s, rank=param['rank'],
                                   init_stdev=param['stdev'],
                                   n_iter=param['n_iter'])
        y_pred_proba[:, n] = fm.fit_predict_proba(X_train, y_train, X_test)
    return y_pred_proba.mean(axis=1)


def balanced_acc(y_test, y_pred_proba, threshold=0.5):
    y_pred = np.ones_like(y_pred_proba)
    y_pred[y_pred_proba < threshold] = -1

    test_n_male = (y_test == -1).sum()
    test_n_female = len(y_test) - test_n_male
    female_ratio = float(test_n_male) / test_n_female
    sample_weight = np.ones_like(y_test)
    sample_weight[y_test == 1] *= female_ratio
    return accuracy_score(y_test, y_pred, sample_weight=sample_weight)


def transform(X_train, y_train, X_test, doc):
    print 'transform'

    if X_test is not None:
        y_test_pred = fit_predict_fm(X_train, y_train, X_test, doc)
        X_test_proba =\
            features.padded_rolling_window(y_test_pred,
                                           doc['transform']['window'])
    else:
        X_test_proba = None

    # cross_validation is need to get predictions for all training samples
    y_train_pred = np.empty_like(y_train, dtype=np.float64)
    y_train_pred.fill(np.nan)
    kf = cross_validation.StratifiedKFold(y_train,
                                          n_folds=doc['transform']['n_folds'],
                                          shuffle=True, random_state=123)
    for i_train, i_test in kf:
        print i_train.shape, i_test.shape
        tmp_pred = fit_predict_fm(X_train[i_train, :],
                                  y_train[i_train], X_train[i_test, :], doc)
        y_train_pred[i_test] = tmp_pred
    X_train_proba = features.padded_rolling_window(y_train_pred,
                                                   doc['transform']['window'])
    return X_train_proba, X_test_proba
