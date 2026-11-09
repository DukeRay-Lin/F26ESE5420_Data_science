"""Model A: the 12 original numerical predictors, scaled inside each fold."""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline():
    return Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])
