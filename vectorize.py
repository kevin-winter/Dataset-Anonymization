from sklearn import preprocessing as pp
import pandas as pd

class Vectorizer():

    def transform(self, df):
        df = df.copy()
        self.columns = df.columns
        self.mappings = {}
        for c in df:
            if df[c].dtype.name.__eq__("category"):
                df[c], self.mappings[c] = df[c].factorize()

        scaler = pp.MinMaxScaler()
        scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        self.scaler = scaler
        return scaled

    def inverse_transform(self, df):
        df = df.copy()
        descaled = pd.DataFrame(self.scaler.inverse_transform(df), columns=self.columns)
        for k, v in self.mappings.items():
            descaled[k] = v[descaled[k].astype(int)]

        return descaled