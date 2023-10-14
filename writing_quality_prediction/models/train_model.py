"""Training and evaluation process of different models."""
# %%

from writing_quality_prediction.data.make_dataset import load_train_data
from writing_quality_prediction.features.build_features import generate_features
import pandas as pd
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


def custom_train_test_split(
    X: pd.DataFrame, y: pd.DataFrame, train_size: float = 0.8, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_train = y.sample(frac=train_size)
    y_test = y.query("id not in @y_train.id")

    X_train = X.query("id in @y_train.id")
    X_test = X.query("id in @y_test.id")

    return X_train, y_train, X_test, y_test


X, y = load_train_data()

# %%
X_agg, y = generate_features(X, y)

# %%
X_train, y_train, X_test, y_test = custom_train_test_split(X=X_agg, y=y)

X_train, X_test = X_train.drop(columns=["id"]), X_test.drop(columns=["id"])
y_train, y_test = y_train.drop(columns=["id"]), y_test.drop(columns=["id"])

# %%
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
pipe = Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])

X_train_transformed = pipe.fit(X_train, y_train)
# %%

cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = pipe.predict(X_test)
scores = cross_validate(
    pipe,
    X_train,
    y_train,
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
    scoring="neg_root_mean_squared_error",
)
