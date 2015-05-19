import features
import numpy as np
from sklearn.utils.testing import assert_array_equal, assert_array_almost_equal
import scipy.sparse as sp


def test_dummy_encoding():
    X = np.arange(3)
    X_sp = features.dummy_encoding(X, 3)
    assert_array_equal(X_sp.todense(), np.identity(3))


def test_neighbors():
    dataset =  {'path': 'data/sample1000/'}
    doc = {'dataset': dataset, 'max_shift': 2, 'shift_features': ['a', 'b']}
    cats = features.cat_matrices(doc['dataset']['path'], is_test=True)
    f_dict = features.neighbors(cats, doc, is_test=False)
    assert 'a1' in f_dict
    assert 'b1' in f_dict


def test_matrix_shift():
    X = sp.csr_matrix(np.arange(8).reshape(4,2))

    X_true = np.array([[2, 4, 8, 4], [3, 6, 10, 5]]).T
    _, _, X_shift1 = features.matrix_shift(X, 1)
    assert_array_equal(X_true, X_shift1.todense())

    X_true = np.array([[6, 0, 0, 0], [7, 0, 0, 1]]).T
    _, _, X_shift1 = features.matrix_shift(X, 3)
    assert_array_equal(X_true, X_shift1.todense())


if __name__ == "__main__":
    test_smooth_encoding()
    test_matrix_shift()
    test_neighbors()
