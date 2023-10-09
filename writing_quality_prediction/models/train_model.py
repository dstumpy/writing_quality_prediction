"""Training and evaluation process of different models."""


from writing_quality_prediction.data.make_dataset import load_train_data
import pandas as pd
from typing import Tuple


def custom_train_test_split(
    X: pd.DataFrame, y: pd.DataFrame, train_size: float = 0.8, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    y_train = y.sample(frac=train_size)
    y_test = y.query("id not in @train_sample.id")

    X_train = X.query("id in @y_train.id")
    X_test = X.query("id in @y_test.id")

    return X_train, y_train, X_test, y_test


X, y = load_train_data()
X_train, y_train, X_test, y_test = custom_train_test_split(X=X, y=y)
