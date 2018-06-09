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
    
    def get_fake(i, text):
        original_idxs = np.array(range(len(text)))
        idxs = original_idxs.copy()
        while True:
            i1, i2 = np.random.choice(idxs, 2, replace=False)
            idxs[i1], idxs[i2] = idxs[i2], idxs[i1]
            sim = sum(idxs == original_idxs)
            length = len(idxs)
            if length*0.6 < sim < length*0.8:
                text = text[idxs]
                break
        nonlocal df
        sys.stdout.write("\r% 5.2f%%"%((i+1)/len(df)*100))
        return text

    np.random.seed(0)
    idxs = df.index

    poses = [get_pos(i, t) for i, t in enumerate(df['text'])]
    df = pd.DataFrame({'pos': poses}, index=idxs)

    df.to_pickle('./data/df_pos_text_%s' % label)
    
    print()

    fakes = [get_fake(i, t) for i, t in enumerate(df['pos'])]
    df = pd.DataFrame({'pos': fakes}, index=idxs)

    df.to_pickle('./data/df_pos_fake_%s' % label)


if __name__ == "__main__":

    print("[tag.py] Loading data ...")
    lable2df = load_df(40000)

    for label in LABELS:
        print("[tag.py] Saving pos dataframe %s ..." % label)
        save_df_pos(lable2df[label], label)
        print()