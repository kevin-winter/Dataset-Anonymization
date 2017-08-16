from sklearn import preprocessing as pp
import pandas as pd
from collections import defaultdict
import numpy as np
from LabelEncoder import LabelEncoder

class Vectorizer:

    def __init__(self, binary=False, feature_range=(0, 1)):
        self.binary = binary
        self.feature_range = feature_range

    def fit_transform(self, df, scale=True, encode=True):
        df = df.copy()
        try:
            self.dtypes  = df.dtypes.to_dict()
            self.columns = df.columns
            self.catcols = df.columns[df.dtypes.astype(str) == "category"]
            self.categories = {col: df[col].cat.categories for col in df[self.catcols]}

            if encode:
                if self.binary:
                    df = pd.get_dummies(df)
                else:
                    self.le = defaultdict(LabelEncoder)
                    df[self.catcols] = df[self.catcols].apply(lambda x: self.le[x.name]
                                                              .fit(pyramid_sorted_categories(x), labels=True)
                                                              .transform(x))
            if scale:
                self.mms = pp.MinMaxScaler(feature_range=self.feature_range)
                df[df.columns] = self.mms.fit_transform(df)

            self.columns_t = df.columns

        except AttributeError:
            print("No Dataframe given. Only scaling possible.")
            if scale:
                self.mms = pp.MinMaxScaler(feature_range=self.feature_range)
                df = self.mms.fit_transform(df)

        return df

    def transform(self, df, scale=True, encode=True):
        df = df.copy()

        try:
            df.columns = self.columns

            if encode:
                if self.binary:
                    df = pd.get_dummies(df)
                else:
                    df[self.catcols] = df[self.catcols].apply(lambda x: self.le[x.name].transform(x))
            if scale:
                df[df.columns] = self.mms.transform(df)
        except AttributeError:
            print("No Dataframe given. Only scaling possible.")
            if scale:
                df = self.mms.transform(df)

        return df

    def inverse_transform(self, df, descale=True, decode=True, clip=None):
        if clip:
            df = np.clip(df, clip[0], clip[1])
        try:
            df = pd.DataFrame(df, columns=self.columns_t)
            if descale:
                df[self.columns_t] = self.mms.inverse_transform(df)

            if decode:
                if self.binary:
                    df = from_dummies(df, categories=self.catcols)
                else:
                    df[self.catcols] = df[self.catcols].astype(int).apply(lambda x: self.le[x.name].inverse_transform(x))

            df = df.astype(self.dtypes)
            for col in df[self.catcols]:
                df[col].cat.set_categories(self.categories[col], inplace=True)

        except AttributeError:
            print("No Dataframe given. Only descaling possible.")
            if descale:
                df = self.mms.inverse_transform(df)

        return df


def from_dummies(data, categories, prefix_sep='_'):
    out = data.copy()
    for l in categories:
        cols, labs = [[c.replace(x,"") for c in data.columns if l+prefix_sep in c] for x in ["", l+prefix_sep]]
        out[l] = pd.Categorical(np.array(labs)[np.argmax(data[cols].as_matrix(), axis=1)])
        out.drop(cols, axis=1, inplace=True)
    return out


def pyramid_sorted_categories(series):
    s = series.value_counts()
    return np.array(s[0::2][::-1].append(s[1::2]).index)