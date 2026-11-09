"""Model B: fold-local one-hot encoding, numerical scaling, and Ridge."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .prepare_data import FEATURES

CATEGORICAL = ['hr', 'weekday', 'mnth', 'season', 'weathersit']
NUMERICAL = [name for name in FEATURES if name not in CATEGORICAL]
# Fixed before viewing validation results; ties select the smaller alpha.
ALPHAS = np.logspace(-4, 4, 25)


def build_pipeline(alpha):
    preprocessing = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ('scale', StandardScaler(), NUMERICAL),
    ], remainder='drop')
    # Dense SVD is deterministic and handles collinear one-hot columns.
    return Pipeline([('preprocess', preprocessing),
                     ('model', Ridge(alpha=float(alpha), solver='svd'))])
