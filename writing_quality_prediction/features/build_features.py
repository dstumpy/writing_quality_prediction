"""Module to generate features for training process."""
import pandas as pd
from typing import Tuple


def generate_features(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_agg = (
        X.groupby("id")
        .aggregate(
            max_events=("event_id", "max"),
            min_down_time=("down_time", "min"),
            max_down_time=("down_time", "max"),
            mean_down_time=("down_time", "mean"),
            std_down_time=("down_time", "std"),
            sum_down_time=("down_time", "sum"),
            min_up_time=("up_time", "min"),
            max_up_time=("up_time", "max"),
            mean_up_time=("up_time", "mean"),
            sum_up_time=("up_time", "sum"),
            std_up_time=("up_time", "std"),
            min_action_time=("action_time", "min"),
            max_action_time=("action_time", "max"),
            mean_action_time=("action_time", "mean"),
            std_action_time=("action_time", "std"),
            median_action_time=("action_time", "median"),
            max_word_count=("word_count", "max"),
            mean_word_count=("word_count", "mean"),
            median_word_count=("word_count", "median"),
            std_word_count=("word_count", "std"),
        )
        .reset_index()
        .sort_values(by="id")
    )

    y_aligned = y.sort_values(by="id")

    return X_agg, y_aligned
