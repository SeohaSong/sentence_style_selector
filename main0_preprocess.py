import pandas as pd
import re
import numpy as np
import sys


def load_data():

    df = pd.read_csv("./data/watcha_movie_review.csv")

    return df


def process_df(df):

    df = df.copy()
    df = df[df['point'] > 0]

    def process(i, text, skip=False):
        text = str(text)
        text = re.sub(r'[^\w,.!?\s]', ' ', text)
        text = re.sub(r'\d+', '0 ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\d+\s)+', '0 ', text)
        if not 50 < len(text) <= 100:
            skip = True
        if not skip and re.compile(r'[a-zA-Zㄱ-ㅎㅏ-ㅣ]').search(text):
            skip = True
        if not skip and re.compile(r'\w{10}').search(text):
            skip = True
        if not skip and re.compile(r'(.+?)\1{3}').search(text):
            skip = True
        elif not skip:
            text = re.sub(r'[.,!?]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'^ ', '', text)
            text = re.sub(r' $', '', text)
        else:
            text = ''
        sys.stdout.write("\r% 5.2f%%"%((i+1)/len(df)*100))
        return text
        
    texts = [process(i, t) for i, t in enumerate(df['text'])]
    print()
    df['text'] = texts
    df = df[df['text'] != '']    
    df = df.reset_index(drop=True)
    
    return df


def get_boundary(df):

    keys = sorted(list(set(df['point'])))
    p2c = {key: len(df[df['point'] == key]) for key in keys}

    def calc_loss(bound):
        b0, b1 = keys.index(bound[0]), keys.index(bound[1])
        counts = [p2c[key] for key in keys]
        l0, l1, l2 = counts[:b0], counts[b0:b1], counts[b1:]
        merged_counts = [sum(l0), sum(l1), sum(l2)]
        loss = np.std(merged_counts)
        return loss, merged_counts

    bounds = [[keys[i], keys[j]]
              for i in range(len(keys))
              for j in range(i+1, len(keys))]
    boundary = min(bounds, key=calc_loss)

    return boundary


def concat_lebel(df, boundary):
    
    def get_label(i, p):
        if p < boundary[0]:
            label = 'neg'
        elif p < boundary[1]:
            label = 'neu' 
        else:
            label = 'pos'
        sys.stdout.write("\r% 5.2f%%"%((i+1)/len(df)*100))
        return label

    label = [get_label(i, p) for i, p in enumerate(df['point'])]
    print()
    df['label'] = label

    return df


def save_df(df):

    df.to_pickle("./data/df_watcha")


if __name__ == "__main__":

    print("[preprocess.py] Loading data ...")
    df = load_data()
    print("[preprocess.py] Processing data ...")
    df = process_df(df)
    print("[preprocess.py] Calculating boundary ...")
    boundary = get_boundary(df)
    print("[preprocess.py] Concatenating label ...")    
    df = concat_lebel(df, boundary)
    print("[preprocess.py] Saving ...")
    save_df(df)
