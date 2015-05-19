from pandas import DataFrame
import blocks
import load_data
import numpy as np
from numpy.testing import assert_array_equal


def test_purity():
    ids = range(10)
    labels = [1, 1, 1, -1, -1, -1, 1, 1, -1, -1]
    blocks_ = [1, 1, 1,  2,  2,  3, 3, 4,  4,  4]
    df = DataFrame({'ids': ids, 'label': labels, 'block': blocks_})
    res = blocks.purity(df).set_index('block')
    assert res.loc[1].miss == 0
    assert res.loc[4].miss == 1
    print res.mean()


def test_recognizer():
    path = 'data/sample1000/'
    df = load_data.data_df(path).head(50)
    blocks_ = np.zeros(len(df))
    blocks_[2:45] = 1
    blocks_[45:] = 2
    df_true_blocks = DataFrame({'session_id': df.session_id,
                                'block': blocks_})
    df_rec_blocks = blocks.recognizer(df)

    assert_array_equal(df_true_blocks.block.values,
                       df_rec_blocks.block.values)


def test_postprocess():
    ids = np.arange(20)

    blocks_ = np.empty(20)
    blocks_[0:2] = 0
    blocks_[2:5] = 1
    blocks_[5:10] = 2
    blocks_[10:20] = 3

    proba = np.empty(20, dtype=np.float)
    proba[0:2] = np.random.uniform(low=.8, high=.9, size=2)
    proba[2:5] = np.random.uniform(low=.01, high=.99, size=3)
    proba[5:10] = np.random.uniform(low=.8, high=.9, size=5)
    proba[10:20] = np.random.uniform(low=.1, high=.3, size=10)

    exp_proba = np.empty(20, dtype=np.float)
    exp_proba[0:2] = proba[0:2]
    exp_proba[2:5] = proba[2:5]
    exp_proba[5:10] = 1
    exp_proba[10:20] = 0

    df = DataFrame({'session_id': ids,
                    'block': blocks_,
                    'proba': proba})


    df_processed = blocks.postprocess(df, min_size=3, lower_proba=.4,
                                      upper_proba=.7)
    assert_array_equal(df_processed.proba.values, exp_proba)
    #print df
    #print df_processed


if __name__ == "__main__":
    test_postprocess()
    test_purity()
    test_recognizer()
