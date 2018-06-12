import pandas as pd
import numpy as np
import sys
import os

from gensim.models import Word2Vec


def load_df():
    
    def get_df(label):
        df = pd.read_pickle("./data/df_pos_%s" % label)
        labels = [label for _ in range(len(df))]
        df['label'] = labels
        return df

    df_watcha = pd.read_pickle('./data/df_watcha')
    labels = sorted(list(set(df_watcha['label'])))
    dfs = [get_df(label) for label in labels]
    df = pd.concat(dfs, axis=0)
    
    return df


def get_words(df):
    
    def get_a(i, pos):
        a = np.array(["-".join(word) for word in pos])
        sys.stdout.write("\r% 5.2f%%"%((i+1)/len(df)*100))
        return a
    
    df['pos'] = [get_a(i, p) for i, p in enumerate(df['pos'])]
    print()

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
        len(words)+1,
        len(model.wv[words[0]])
    ])

    for i, word in enumerate(words):
        lookup_table[i] = model.wv[word]
        sys.stdout.write("\r% 5.2f%%"%((i+1)/len(words)*100))
    print()

    lookup_table[-1] = np.zeros(len(model.wv[words[0]]))

    pd.to_pickle(lookup_table, "./data/lookup_table")


def save_df(model, df):

    words = model.wv.index2word

    def get_true(i, ws):
        idxs = np.array([words.index(w) for w in ws])
        sys.stdout.write("\r% 5.2f%%" % ((i+1)/len(df)*100))
        return idxs

    def get_fake(i, pos):
        fake = pos.copy()
        len_ = len(fake)
        idxs = list(range(len_))
        while True:
            a, b = np.random.choice(idxs, 2, replace=False)
            fake[a], fake[b] = fake[b], fake[a]
            if sum(pos == fake) < len_*0.7:
                break
        sys.stdout.write("\r%5.2f%%" % ((i+1)/len(df)*100))
        return fake
    
    print('(Making true case)')
    df['pos'] = [get_true(i, ws) for i, ws in enumerate(df['pos'])]
    print()

    df_fake = df.copy()
    print('(Making fake case)')
    df_fake['pos'] = [get_fake(i, p) for i, p in enumerate(df['pos'])]
    print()

    df['valid'] = ['real' for _ in range(len(df))]
    df_fake['valid'] = ['fake' for _ in range(len(df))]
    
    df = pd.concat([df, df_fake], axis=0)

    pd.to_pickle(df, "./data/df")


if __name__ == "__main__":

    print("[embedding.py] Loading data ...")
    df = load_df()
    print("[embedding.py] Merging word and pos ...")    
    df = get_words(df)
    print("[embedding.py] Saving embedding model ...")   
    save_w2v_model(df)
    print("[embedding.py] Loading embedding model ...")
    model = load_model()
    print("[embedding.py] Saving lookup table ...")
    save_lookup_table(model)
    print("[embedding.py] Saving final dataframe ...")
    save_df(model, df)
    