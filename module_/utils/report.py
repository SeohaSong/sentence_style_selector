import pandas as pd
from matplotlib import pyplot as plt

from preprocess import load_data, get_meta


class Report():

    def __init__(self):

        self.df_raw = load_data()
        self.df = pd.read_pickle("./data/df_watcha")

    def present(self):

        df, df_raw = self.df, self.df_raw

        boundary, counts, label2count = get_meta(df)

        print("총 리뷰 수:\t", len(df_raw))
        lengths = [len(sen) for sen in df_raw["text"] if type(sen) == str]
        plt.plot(lengths)
        plt.show()
        length2count = {l: l for l in list(set(lengths))}
        for key in lengths:
            length2count[key] += 1
        keys = sorted(list(length2count))
        plt.bar(keys, [length2count[key] for key in keys])
        plt.show()

        print("전처리 후 리뷰 수:\t", len(df))
        keys = sorted(list(label2count))
        for key in keys:
            print("\t%.1f 점 리뷰 수:\t" % key, label2count[key])
            
        lengths = [len(sen) for sen in df["text"]]
        plt.plot(lengths)
        plt.show()
        length2count = {l: l for l in list(set(lengths))}
        for key in lengths:
            length2count[key] += 1
        keys = sorted(list(length2count))
        plt.bar(keys, [length2count[key] for key in keys])
        plt.show()

        print("최적 경계선:\t", boundary)
        print("경계별 리뷰 수")
        names = ['부정', '중립', '긍정']
        for i in range(len(counts)):
            print("\t%s:\t%s" % (names[i], counts[i]))

        keys = sorted(list(label2count))
        plt.bar(keys, [label2count[key] for key in keys])
        plt.show()
