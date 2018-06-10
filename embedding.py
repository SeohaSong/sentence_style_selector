import pandas as pd
import numpy as np
import sys
import os

from gensim.models import Word2Vec


def load_df(n_batch=None):
    
    def get_df(label):
        nonlocal n_batch
        df = pd.read_pickle("./data/df_pos_%s" % label)
        se_label = pd.Series([label for _ in range(len(df))],
                             index=df.index,
                             name='label')
        df = pd.concat([df, se_label], axis=1)
        if n_batch:
            np.random.seed(0)
            idxs = np.random.choice(df.index, n_batch, replace=False)
            df = df.loc[idxs]
        return df

    labels = ['pos', 'neu', 'neg']
    dfs = [get_df(label) for label in labels]
    df = pd.concat(dfs, axis=0)
    
    return df


def get_words(df):
    
    def get_words(i, pos):
        words = np.array(["-".join(word) for word in pos])
        nonlocal df
        sys.stdout.write("\r% 5.2f%%"%((i+1)/len(df)*100))
        return words
    
    se_words = pd.Series(
        [get_words(i, p) for i, p in enumerate(df['pos'])],
        index=df.index
    )
    df['pos'] = se_words

    return df


def save_w2v_model(df):
    
    n_size = 512
    n_winow = 8
    min_count = 1
    workers = os.cpu_count()
    sens = [sen.tolist() for sen in df['pos']]

    model = Word2Vec(
        sens,
        size=n_size,
        window=n_winow,
        min_count=min_count,
        workers=workers,
        iter=100,
    )
    
    model.save("./data/w2v_model")


def load_model():
    model = Word2Vec.load("./data/w2v_model")
    return model


def save_lookup_table(model):
    
    words = model.wv.index2word
    lookup_table = np.zeros([
        len(words),
        len(model.wv[words[0]])
    ])

    for i, word in enumerate(words):
        lookup_table[i] = model.wv[word]
        sys.stdout.write("\r% 5.2f%%"%((i+1)/len(words)*100))

    pd.to_pickle(lookup_table, "./data/lookup_table")


def save_df(model, df):

    def get_idxs(i, ws):
        nonlocal words
        idxs = np.array([words.index(w) for w in ws])
        nonlocal se_words
        sys.stdout.write("\r% 5.2f%%" % ((i+1)/len(se_words)*100))
        return idxs

    words = model.wv.index2word
    se_words = df['pos']
    se_idxs = pd.Series(
        [get_idxs(i, ws) for i, ws in enumerate(se_words)],
        index=df.index
    )

    df['pos'] = se_idxs
    pd.to_pickle(df, "./data/df")


if __name__ == "__main__":

    print("[embedding.py] Loading data ...")
    df = load_df()
    print("[embedding.py] Merging word and pos ...")    
    df = get_words(df)
    print()
    print("[embedding.py] Saving embedding model ...")   
    save_w2v_model(df)

    print("[embedding.py] Loading embedding model ...")
    model = load_model()
    print("[embedding.py] Saving lookup table ...")
    save_lookup_table(model)
    print()
    print("[embedding.py] Saving final dataframe ...")
    save_df(model, df)
    print()