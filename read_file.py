import pandas as pd
from vectorize import Vectorizer

headers = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship",
           "race","sex","capital-gain","capital-loss","hours-per-week","native-country","salary"]

dtypes = {"age": int,
          "workclass": "category",
          "fnlwgt": int,
          "education": "category",
          "education-num": int,
          "marital-status": "category",
          "occupation": "category",
          "relationship": "category",
          "race": "category",
          "sex": "category",
          "capital-gain": int,
          "capital-loss": int,
          "hours-per-week": int,
          "native-country": "category",
          "salary": "category"}

df = pd.read_csv("adult.txt", names=headers, dtype=dtypes)
vec = Vectorizer()
dft = vec.transform(df)
dfo = vec.inverse_transform(dft)
print(dfo)
print(df.equals(dfo))