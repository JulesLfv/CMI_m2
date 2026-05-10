import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae_lat = np.mean(np.abs(err[:, 0]))
    mae_lon = np.mean(np.abs(err[:, 1]))
    rmse_lat = np.sqrt(np.mean(err[:, 0] ** 2))
    rmse_lon = np.sqrt(np.mean(err[:, 1] ** 2))
    euclid = np.sqrt(np.sum(err ** 2, axis=1))
    hav = haversine_km(y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1])

    return {
        "mae_lat": float(mae_lat),
        "mae_lon": float(mae_lon),
        "rmse_lat": float(rmse_lat),
        "rmse_lon": float(rmse_lon),
        "euclidean_mean_deg": float(np.mean(euclid)),
        "haversine_mean_km": float(np.mean(hav)),
    }
