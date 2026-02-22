"""
Shared data loading and split utilities for UB Keystroke Dynamics.
Supports: single session-based split, session rotation (3-fold), and cross-condition (baseline vs rotation).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Default path when running from src/
DEFAULT_CSV = "../outputs/keystroke_features.csv"
META_COLS = ["user", "session", "condition", "file"]
SESSIONS = ["s0", "s1", "s2"]


def load_keystroke_data(csv_path=DEFAULT_CSV, condition=None, fillna=0):
    """Load feature CSV and optionally filter by condition ('baseline', 'rotation', or None for both)."""
    df = pd.read_csv(csv_path)
    df = df.fillna(fillna)
    if condition is not None:
        df = df[df["condition"] == condition]
    return df


def get_feature_columns(df):
    """Return list of feature columns (exclude meta)."""
    return [c for c in df.columns if c not in META_COLS]


def get_single_split(df, train_sessions=("s0", "s1"), test_session="s2"):
    """
    Single session-based split: train on train_sessions, test on test_session.
    Returns (X_train, X_test, y_train, y_test, label_encoder).
    """
    feature_cols = get_feature_columns(df)
    train_df = df[df["session"].isin(list(train_sessions))]
    test_df = df[df["session"] == test_session]

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["user"])
    y_test = le.transform(test_df["user"])

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    return X_train, X_test, y_train, y_test, le


def iter_session_rotation_splits(df):
    """
    Session rotation: 3-fold leave-one-session-out.
    Each fold uses one session as test and the other two as train.
    Yields (X_train, X_test, y_train, y_test, test_session_name) per fold.
    """
    feature_cols = get_feature_columns(df)
    for test_sess in SESSIONS:
        train_sessions = [s for s in SESSIONS if s != test_sess]
        train_df = df[df["session"].isin(train_sessions)]
        test_df = df[df["session"] == test_sess]

        le = LabelEncoder()
        y_train = le.fit_transform(train_df["user"])
        y_test = le.transform(test_df["user"])

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        yield X_train, X_test, y_train, y_test, test_sess


def get_cross_condition_split(df_all, train_condition="baseline", test_condition="rotation"):
    """
    Cross-condition split: train on one condition (e.g. baseline), test on another (e.g. rotation).
    Uses all sessions for both train and test (different condition).
    Returns (X_train, X_test, y_train, y_test, label_encoder).
    """
    feature_cols = get_feature_columns(df_all)
    train_df = df_all[df_all["condition"] == train_condition]
    test_df = df_all[df_all["condition"] == test_condition]

    if train_df.empty or test_df.empty:
        return None

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["user"])
    # Only include test users that were seen in training
    test_df = test_df[test_df["user"].isin(le.classes_)]
    if test_df.empty:
        return None
    y_test = le.transform(test_df["user"])

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    return X_train, X_test, y_train, y_test, le
