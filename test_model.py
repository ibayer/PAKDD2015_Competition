import model
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
import scipy.sparse as sp


doc = {'dataset': {'path': 'data/sample1000/'},
       'features': ['a', 'b'],
       'fm_param': {'n_iter': 350, 'stdev': .001, 'rank': 4},
       'seeds': [123, 345],
       'output': {},
       'threshold': .84,
       'shift_features': ['a', 'b'],
       'max_shift': 5,
       'min_freq': [1, 5, 10, 20],
       'min_size': 10,
       'lower_proba': .7,
       'upper_proba': .9,
       'submission_path': 'data/full/submission/submission.txt'
       }


def _test_submission():
    import tempfile
    temp = tempfile.NamedTemporaryFile()
    o_path = temp.name
    doc['submission_path'] = o_path

    model.submission(doc)
    pred = {11: [1],
            22: [1, 2],
            33: [1, 2,3]
            }
    true_file = "11;1\n22;1,2\n33;1,2,3\n"

    with open(temp.name) as f:
        data = f.read()
        #print data

    assert true_file == data

    for t, w in zip(true_file, data):
        assert t == w

if __name__ == "__main__":
    test_submission()
