from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_clean_data(cfg: Dict) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(cfg["csv_path"])

    date_col = cfg["date_col"]
    storm_col = cfg["storm_id_col"]
    lat_col = cfg["lat_col"]
    lon_col = cfg["lon_col"]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    qa = {
        "n_rows_raw": len(df),
        "missing_dates": int(df[date_col].isna().sum()),
        "missing_lat": int(df[lat_col].isna().sum()),
        "missing_lon": int(df[lon_col].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    df = df.dropna(subset=[date_col, storm_col, lat_col, lon_col]).copy()
    df = df.drop_duplicates().copy()
    df = df.sort_values([storm_col, date_col]).reset_index(drop=True)

    qa["n_rows_clean"] = len(df)
    qa["n_storms"] = int(df[storm_col].nunique())

    dt = df.groupby(storm_col)[date_col].diff().dt.total_seconds().div(3600.0)
    qa["irregular_intervals_pct"] = float((dt.dropna() != 6).mean() * 100)

    return df, qa


def add_engineered_features(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.copy()
    date_col = cfg["date_col"]
    storm_col = cfg["storm_id_col"]
    lat_col = cfg["lat_col"]
    lon_col = cfg["lon_col"]

    df["delta_lat"] = df.groupby(storm_col)[lat_col].diff().fillna(0.0)
    df["delta_lon"] = df.groupby(storm_col)[lon_col].diff().fillna(0.0)
    df["delta_hours"] = (
        df.groupby(storm_col)[date_col].diff().dt.total_seconds().div(3600.0).fillna(6.0)
    )
    df["speed_deg_per_h"] = np.sqrt(df["delta_lat"] ** 2 + df["delta_lon"] ** 2) / df["delta_hours"].clip(lower=1e-6)
    df["heading_rad"] = np.arctan2(df["delta_lat"], df["delta_lon"])

    df["month"] = df[date_col].dt.month.astype(float)
    df["day_of_year"] = df[date_col].dt.dayofyear.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df


def split_by_storm_id(df: pd.DataFrame, storm_col: str, train_ratio=0.7, val_ratio=0.15, seed=42):
    storms = df[storm_col].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    n = len(storms)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_storms = set(storms[:n_train])
    val_storms = set(storms[n_train:n_train + n_val])
    test_storms = set(storms[n_train + n_val:])

    train_df = df[df[storm_col].isin(train_storms)].copy()
    val_df = df[df[storm_col].isin(val_storms)].copy()
    test_df = df[df[storm_col].isin(test_storms)].copy()
    return train_df, val_df, test_df


def build_sequences(df: pd.DataFrame, cfg: Dict, window_size: int, feature_cols: List[str], target_cols: List[str]):
    storm_col = cfg["storm_id_col"]
    min_len = cfg.get("min_sequence_length", window_size + 1)

    X_list, y_list, meta = [], [], []

    for sid, g in df.groupby(storm_col):
        g = g.sort_values(cfg["date_col"]).reset_index(drop=True)
        if len(g) < max(min_len, window_size + 1):
            continue

        vals = g[feature_cols].values.astype(np.float32)
        targets = g[target_cols].values.astype(np.float32)

        for i in range(window_size, len(g)):
            X_list.append(vals[i - window_size:i])
            y_list.append(targets[i])
            meta.append((sid, g.loc[i, cfg["date_col"]]))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y, meta


def fit_scalers(X_train: np.ndarray, y_train: np.ndarray):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    n, t, f = X_train.shape
    X_train_2d = X_train.reshape(-1, f)
    x_scaler.fit(X_train_2d)
    y_scaler.fit(y_train)
    return x_scaler, y_scaler


def apply_scalers(X: np.ndarray, y: np.ndarray, x_scaler: StandardScaler, y_scaler: StandardScaler):
    n, t, f = X.shape
    X_scaled = x_scaler.transform(X.reshape(-1, f)).reshape(n, t, f)
    y_scaled = y_scaler.transform(y)
    return X_scaled.astype(np.float32), y_scaled.astype(np.float32)
