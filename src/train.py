from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import HurricaneWindowDataset
from evaluate import regression_metrics
from models import MLPRegressor, RNNRegressor, TCNRegressor
from preprocessing import (
    add_engineered_features,
    apply_scalers,
    build_sequences,
    fit_scalers,
    load_and_clean_data,
    split_by_storm_id,
)
from utils import EarlyStopping, count_parameters, ensure_dirs, get_device, load_config, set_seed


def train_torch_model(model, train_loader, val_loader, cfg_train, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train["learning_rate"], weight_decay=cfg_train["weight_decay"])
    es = EarlyStopping(patience=cfg_train["early_stopping_patience"])

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")

    for _ in range(cfg_train["epochs"]):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_losses.append(criterion(model(xb), yb).item())

        tr, va = float(np.mean(tr_losses)), float(np.mean(va_losses))
        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        if va < best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if es.step(va):
            break

    model.load_state_dict(best_state)
    return model, history


def predict_model(model, loader, device):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            ys.append(yb.numpy())
    return np.vstack(ys), np.vstack(preds)


def build_model(name, input_dim, window_size, cfg_models):
    if name == "mlp":
        return MLPRegressor(input_dim, window_size, cfg_models["mlp"]["hidden_sizes"], dropout=cfg_models["mlp"]["dropout"])
    if name == "rnn":
        c = cfg_models["rnn"]
        return RNNRegressor(input_dim, c["hidden_size"], c["num_layers"], dropout=c["dropout"], rnn_type="RNN")
    if name == "lstm":
        c = cfg_models["lstm"]
        return RNNRegressor(input_dim, c["hidden_size"], c["num_layers"], dropout=c["dropout"], rnn_type="LSTM")
    if name == "gru":
        c = cfg_models["gru"]
        return RNNRegressor(input_dim, c["hidden_size"], c["num_layers"], dropout=c["dropout"], rnn_type="GRU")
    if name == "tcn":
        c = cfg_models["tcn"]
        return TCNRegressor(input_dim, tuple(c["channels"]), c["kernel_size"], c["dropout"])
    raise ValueError(name)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    ensure_dirs("results/models", "results/metrics", "results/figures")

    df, qa = load_and_clean_data(cfg["data"])
    df = add_engineered_features(df, cfg["data"])
    print("QA:", qa)

    if cfg["data"]["features_mode"] == "basic":
        feature_cols = [cfg["data"]["lat_col"], cfg["data"]["lon_col"]]
    else:
        feature_cols = [cfg["data"]["lat_col"], cfg["data"]["lon_col"], "delta_lat", "delta_lon", "speed_deg_per_h", "heading_rad", "delta_hours", "month_sin", "month_cos", "doy_sin", "doy_cos"]

    target_cols = [cfg["data"]["lat_col"], cfg["data"]["lon_col"]]

    tr_df, va_df, te_df = split_by_storm_id(
        df,
        cfg["data"]["storm_id_col"],
        cfg["data"]["split"]["train_ratio"],
        cfg["data"]["split"]["val_ratio"],
        cfg["seed"],
    )

    all_results = []
    for w in cfg["data"]["window_sizes"]:
        Xtr, ytr, _ = build_sequences(tr_df, cfg["data"], w, feature_cols, target_cols)
        Xva, yva, _ = build_sequences(va_df, cfg["data"], w, feature_cols, target_cols)
        Xte, yte, _ = build_sequences(te_df, cfg["data"], w, feature_cols, target_cols)

        if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
            continue

        x_scaler, y_scaler = fit_scalers(Xtr, ytr)
        Xtr_s, ytr_s = apply_scalers(Xtr, ytr, x_scaler, y_scaler)
        Xva_s, yva_s = apply_scalers(Xva, yva, x_scaler, y_scaler)
        Xte_s, yte_s = apply_scalers(Xte, yte, x_scaler, y_scaler)

        train_loader = DataLoader(HurricaneWindowDataset(Xtr_s, ytr_s), batch_size=cfg["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(HurricaneWindowDataset(Xva_s, yva_s), batch_size=cfg["training"]["batch_size"], shuffle=False)
        test_loader = DataLoader(HurricaneWindowDataset(Xte_s, yte_s), batch_size=cfg["training"]["batch_size"], shuffle=False)

        # Baselines on unscaled targets
        persist_pred = Xte[:, -1, :2]
        mean_disp = Xte[:, -1, :2] + (Xte[:, -1, :2] - Xte[:, -2, :2])
        for bname, pred in [("persistence", persist_pred), ("mean_displacement", mean_disp)]:
            m = regression_metrics(yte, pred)
            m.update({"model": bname, "window": w, "params": 0, "train_seconds": 0.0, "infer_ms_per_sample": 0.0})
            all_results.append(m)

        for model_name in [m for m in cfg["experiment"]["model_names"] if m not in ["persistence", "mean_displacement"]]:
            model = build_model(model_name, len(feature_cols), w, cfg["models"]).to(device)
            t0 = time.time()
            model, hist = train_torch_model(model, train_loader, val_loader, cfg["training"], device)
            t_train = time.time() - t0

            t1 = time.time()
            y_true_s, y_pred_s = predict_model(model, test_loader, device)
            t_inf = (time.time() - t1) * 1000 / len(y_true_s)

            y_true = y_scaler.inverse_transform(y_true_s)
            y_pred = y_scaler.inverse_transform(y_pred_s)
            metrics = regression_metrics(y_true, y_pred)
            metrics.update({
                "model": model_name,
                "window": w,
                "params": count_parameters(model),
                "train_seconds": t_train,
                "infer_ms_per_sample": t_inf,
            })
            all_results.append(metrics)

            torch.save(model.state_dict(), f"results/models/{model_name}_w{w}.pt")

            plt.figure()
            plt.plot(hist["train_loss"], label="train")
            plt.plot(hist["val_loss"], label="val")
            plt.title(f"Loss - {model_name} - w={w}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/figures/loss_{model_name}_w{w}.png")
            plt.close()

    results_df = pd.DataFrame(all_results).sort_values(["haversine_mean_km", "euclidean_mean_deg"])
    results_df.to_csv("results/metrics/summary_metrics.csv", index=False)
    with open("results/metrics/data_quality.json", "w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2)
    print(results_df.head(20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
