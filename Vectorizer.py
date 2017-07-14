from sklearn import preprocessing as pp
import pandas as pd
from collections import defaultdict
import numpy as np

class Vectorizer():

    def __init__(self, binary=False):
        self.binary = binary

    def transform(self, df, scale=True, encode=True):
        df = df.copy()
        self.columns = df.columns
        self.catcols = df.columns[df.dtypes.astype(str) == "category"]

        if encode:
            if self.binary:
                df = pd.get_dummies(df)
            else:
                self.le = defaultdict(pp.LabelEncoder)
                df[self.catcols] = df[self.catcols].apply(lambda x: self.le[x.name].fit_transform(x))

        if scale:
            self.mms = pp.MinMaxScaler()
            df[df.columns] = self.mms.fit_transform(df)

        self.columns_t = df.columns
        return df

    def inverse_transform(self, df, descale=True, decode=True):
        df = pd.DataFrame(df, columns=self.columns_t)
        if descale:
            df[self.columns_t] = self.mms.inverse_transform(df)

        if decode:
            if self.binary:
                df = from_dummies(df, categories=self.catcols)
            else:
                df[self.catcols] = df[self.catcols].astype(int).apply(lambda x: self.le[x.name].inverse_transform(x))

        return df


def from_dummies(data, categories, prefix_sep='_'):
    out = data.copy()
    for l in categories:
        cols, labs = [[c.replace(x,"") for c in data.columns if l+prefix_sep in c] for x in ["", l+prefix_sep]]
        out[l] = pd.Categorical(np.array(labs)[np.argmax(data[cols].as_matrix(), axis=1)])
        out.drop(cols, axis=1, inplace=True)
    return out