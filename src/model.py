"""Preprocessing + SMOTE + classifier, with SMOTE applied to the TRAINING data only."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CAT_FEATURES, NUM_FEATURES
from .smote import oversample


def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=15, sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, NUM_FEATURES), ("cat", cat, CAT_FEATURES)])


class RiskModel:
    def __init__(self, estimator, use_smote=True, random_state=42):
        self.estimator = estimator
        self.use_smote = use_smote
        self.random_state = random_state
        self.pre = build_preprocessor()
        self.smote_backend = None

    def fit(self, X, y):
        Xt = self.pre.fit_transform(X)
        y = np.asarray(y)
        if self.use_smote:
            Xt, y, self.smote_backend = oversample(Xt, y, self.random_state)
        self.estimator.fit(Xt, y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(self.pre.transform(X))[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def feature_names(self):
        return list(self.pre.get_feature_names_out())
