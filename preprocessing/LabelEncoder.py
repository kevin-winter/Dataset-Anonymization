import numpy as np


class LabelEncoder:

    def fit(self, y, labels=False):
        if labels:
            self.classes_ = np.array(y)
        else:
            self.classes_ = np.unique(y)
        return self

    def fit_transform(self, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        return y

    def transform(self, y):
        classes = np.unique(y)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            raise ValueError("y contains new labels: %s" % str(diff))
        return np.where(self.classes_ == np.asarray(y)[:, None])[1]

    def inverse_transform(self, y):
        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if len(diff) != 0:
            raise ValueError("y contains new labels: %s" % str(diff))
        y = np.asarray(y)
        return self.classes_[y]