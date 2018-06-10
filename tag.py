import pandas as pd
import sys
import numpy as np

import konlpy


LABELS = ['pos', 'neu', 'neg']


def load_df(n_batch=None):

    def get_df(label):
        nonlocal df, n_batch
        df_ = df[df['label'] == label]
        if n_batch:
            np.random.seed(0)
            idxs = np.random.choice(df_.index, n_batch, replace=False)
            df_ = df_.loc[idxs]
        return df_

    global LABELS
    df = pd.read_pickle("./data/df_watcha")
    label2df = {label: get_df(label) for label in LABELS}

    return label2df


def save_df_pos(df, label):

    def get_pos(i, text):
        pos = konlpy.tag.Kkma().pos(text)
        nonlocal df
        sys.stdout.write("\r% 5.2f%%"%((i+1)/len(df)*100))
        return np.array(pos)

    np.random.seed(0)
    idxs = df.index

    poses = [get_pos(i, t) for i, t in enumerate(df['text'])]
    df = pd.DataFrame({'pos': poses}, index=idxs)

    df.to_pickle('./data/df_pos_%s' % label)


if __name__ == "__main__":

    print("[tag.py] Loading data ...")
    lable2df = load_df()

    for label in LABELS:
        print("[tag.py] Saving pos dataframe %s ..." % label)
        save_df_pos(lable2df[label], label)
        print()
