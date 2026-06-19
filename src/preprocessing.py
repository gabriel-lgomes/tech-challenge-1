import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = 'diagnosis'
DROP_COLS = ['id', 'Unnamed: 32']


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET] = df[TARGET].map({'B': 0, 'M': 1})
    return df


def remove_high_correlation(df: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list]:
    """Remove uma de cada par de features com correlação acima do threshold."""
    corr = df.drop(columns=[TARGET]).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    return X_train_scaled, X_test_scaled, scaler
