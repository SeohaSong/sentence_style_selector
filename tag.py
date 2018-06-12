import pandas as pd
import sys
import numpy as np

import konlpy


def load_df():

    df = pd.read_pickle("./data/df_watcha")
    
    def get_df(label):
        df_ = df[df['label'] == label]
        return df_
    
    labels = sorted(list(set(df['label'])))
    label2df = {label: get_df(label) for label in labels}

    return label2df, labels


def save_df_pos(label2df, labels):

    def get_pos(i, text):
        pos = konlpy.tag.Mecab().pos(text)
        sys.stdout.write("\r% 5.2f%%"%((i+1)/len(df)*100))
        return np.array(pos)

    for label in labels:
        print('(Saving %s)' % label)
        df = label2df[label]
        poses = [get_pos(i, t) for i, t in enumerate(df['text'])]
        df = pd.DataFrame({'pos': poses}, index=df.index)
        df.to_pickle('./data/df_pos_%s' % label)
        print()


if __name__ == "__main__":

    print("[tag.py] Loading data ...")
    lable2df, labels = load_df()

    print("[tag.py] Saving pos dataframe ...")
    save_df_pos(lable2df, labels)
