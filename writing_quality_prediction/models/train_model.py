"""Training and evaluation process of different models."""
# %%

from writing_quality_prediction.data.make_dataset import load_train_data
from writing_quality_prediction.features.build_features import (
    generate_features,
    preprocess_data,
)
from writing_quality_prediction.models.model_selection import custom_train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


X, y = load_train_data()

ohe_features = ["activity", "down_event", "up_event", "text_change"]
oe_features = ["id"]

# %%
# X = preprocess_data(X=X, oe_features=oe_features, ohe_features=ohe_features)

# %%
X_agg, y = generate_features(X, y, ohe_features=ohe_features)

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

# %%
# Hyperparameter tuning


from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        "n_estimators": [
            100,
            500,
            1000,
        ],
        "max_features": [1],
        "max_depth": [1, 2],
        "min_samples_split": [2, 8],
        "min_samples_leaf": [2, 6],
        "bootstrap": [False],
    },
]

model = RandomForestRegressor()
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    verbose=2,
    return_train_score=True,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# %%
