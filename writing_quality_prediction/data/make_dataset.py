"""Load raw data."""

from writing_quality_prediction.config.paths import DATA_DIR
import pandas as pd
from typing import Tuple

def load_train_data() -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_logs = pd.read_csv(DATA_DIR / "raw" / "train_logs.csv")
    train_scores = pd.read_csv(DATA_DIR / "raw" / "train_scores.csv")

    return train_logs, train_scores


