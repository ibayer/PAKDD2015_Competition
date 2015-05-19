import model


doc = {'dataset': {'path': 'data/sample1000/'},
       # 'dataset': {'path': 'data/full/'},
       'features': ['a', 'a1', 'a1-2', 'b', 'b1-3', 'd', 'd1'],
       'fm_param': {'n_iter': 350, 'stdev': .001, 'rank': 4},
       'seeds': [123, 345, 231, 500, 442, 873, 921, 111, 222],
       'output': {},
       'threshold': .84,
       'shift_features': ['a', 'b', 'd'],
       'max_shift': 5,
       'min_freq': [1, 5, 10, 20],
       'min_size': 10,
       'lower_proba': .7,
       'upper_proba': .9,
       'submission_path': 'data/full/submission/submission.txt'
       }

print model.evaluate(doc)
#model.submission(doc)
