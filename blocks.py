import numpy as np
import load_data


def purity(df):
    """
    df:
    columns = [session_id, label, block]

    return: df
        columns = [block, miss]
    """
    records = []
    for id_, group in df.groupby('block'):
        score = 0 if group.label.nunique() == 1 else np.nan
        # print group, 'unique', group.label.nunique()
        if not score == 0:
            score = group.groupby('label').size().min()
        records.append((id_, score, len(group)))
    return df.from_records(records, columns=['block', 'miss', 'size'])


def get_blocks(doc, is_test=False):
    path = doc['dataset']['path']
    df = load_data.data_df(path, is_test)
    return recognizer(df[['session_id', 'start', 'end']])


def recognizer(df):
    """
    columns = ['session_id', 'start', 'end']
    return df: columns = [session_id, block]
    """
    df['block'] = (df.end.shift(1) > df.end).astype(int).cumsum()
    return df


def postprocess(df, min_size, lower_proba=.5, upper_proba=.85):
    """
    columns: ['session_id', 'proba', 'block']
    return df:
            columns: ['session_id', 'proba', 'block']
    """
    def block_smoothing(group):
        if len(group) > min_size:
            mean_proba = np.median(group.proba)
            if mean_proba >= upper_proba:
                group.proba = 1
            elif mean_proba < lower_proba:
                group.proba = 0
        return group
    return df.groupby('block').apply(block_smoothing)
