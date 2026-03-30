import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

####################################################
# Utils for use by game_model.py and lap_model.py. #
####################################################

def build_track_ground_truth(
    laps: pd.DataFrame,
    track: str,
    cl: pd.DataFrame,
    bin_m: float = 5.0,
) -> pd.DataFrame:
    """
    Distance-domain template for one track:
    expected x/y, curvature, speed, brake likelihood.
    """
    d = laps.loc[laps["track"].astype(str) == str(track)].copy()
    if d.empty:
        return pd.DataFrame()

    brake_col = "y_brake_zone"; throttle_col = "y_throttle_zone"

    # Project training laps to centreline for accuracy/consistency:
    projected = project_to_centreline(laps[laps["track"] == track], cl)
    projected["cl_bin"] = (projected["cl_dist"] / bin_m).round().astype(int) * bin_m

    gt = (
        projected.groupby("cl_bin", as_index=False)
         .agg(
             cl_dist=("cl_bin", "first"),
             x_exp=("x", "mean"),
             y_exp=("y", "mean"),
             c_exp=("c_smooth", "mean"),
             norm_speed_exp=("norm_speed", "mean"),
             speed_exp=("speed", "mean"),
             p_brake_exp=(brake_col, "mean"),
             p_throttle_exp=(throttle_col, "mean"),
             brake_exp=("brake", "mean"),      
             throttle_exp=("throttle", "mean"),
         )
         .sort_values("cl_dist")
         .reset_index(drop=True)
    )
    gt["c_exp_ahead"] = gt["c_exp"].shift(-3).fillna(gt["c_exp"])
    return gt

def build_centreline(laps: pd.DataFrame, track: str, bin_m: float = 5.0) -> pd.DataFrame:
    """
    Produces a canonical centreline for the track: evenly-spaced x/y points
    with their cumulative arc-length as 'cl_dist'.
    """
    d = laps.loc[laps["track"].astype(str) == str(track)].copy()
    if d.empty:
        return pd.DataFrame()

    d["dist_bin"] = (d["distance"].astype(float) / bin_m).round().astype(int) * bin_m
    cl = (
        d.groupby("dist_bin", as_index=False)
         .agg(x=("x", "mean"), y=("y", "mean"))
         .sort_values("dist_bin")
         .reset_index(drop=True)
    )
    cl["x"] = cl["x"].rolling(window=11, center=True, min_periods=1).mean()
    cl["y"] = cl["y"].rolling(window=11, center=True, min_periods=1).mean()

    dx = cl["x"].diff().fillna(0.0)
    dy = cl["y"].diff().fillna(0.0)
    cl["cl_dist"] = np.sqrt(dx**2 + dy**2).cumsum()
    return cl

def project_to_centreline(lap_df: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    cl_xy = cl[["x", "y"]].to_numpy()
    tree = cKDTree(cl_xy)

    lap_xy = lap_df[["x", "y"]].to_numpy()
    _, idx = tree.query(lap_xy)

    cl_d = cl["cl_dist"].to_numpy()
    cl_x = cl["x"].to_numpy()
    cl_y = cl["y"].to_numpy()

    proj_d = cl_d[idx]

    fwd_j = np.clip(idx + 1, 0, len(cl_x) - 1)
    bwd_j = np.clip(idx - 1, 0, len(cl_x) - 1)
    fwd = np.stack([cl_x[fwd_j] - cl_x[bwd_j], cl_y[fwd_j] - cl_y[bwd_j]], axis=1).astype(float)
    off = lap_xy - cl_xy[idx]
    lateral = (fwd[:, 0] * off[:, 1] - fwd[:, 1] * off[:, 0]) / (np.linalg.norm(fwd, axis=1) + 1e-9)

    out = lap_df.copy()
    out["cl_dist"] = proj_d
    out["lateral_err"] = lateral
    return out

class Curvature:
    @staticmethod
    def get_curvature(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        n = len(x)
        kappa = np.zeros(n, dtype=float)
        if n < 3:
            return kappa

        for i in range(1, n - 1):
            A = (x[i - 1], y[i - 1])
            B = (x[i], y[i])
            C = (x[i + 1], y[i + 1])
            kappa[i] = Curvature.calc_curvature(A, B, C)

        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]
        return kappa

    def calc_curvature(A, B, C):
        Ax, Ay = float(A[0]), float(A[1])
        Bx, By = float(B[0]), float(B[1])
        Cx, Cy = float(C[0]), float(C[1])

        if not (np.isfinite(Ax) and np.isfinite(Ay) and np.isfinite(Bx) and np.isfinite(By) and np.isfinite(Cx) and np.isfinite(Cy)):
            return 0.0

        v1 = np.array([Bx - Ax, By - Ay], dtype=float)
        v2 = np.array([Cx - Bx, Cy - By], dtype=float)

        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        ds = (n1 + n2) / 2.0
        if not np.isfinite(ds) or ds <= 1e-9:
            return 0.0

        angle1 = float(np.arctan2(v1[1], v1[0]))
        angle2 = float(np.arctan2(v2[1], v2[0]))

        d_theta = (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi
        if not np.isfinite(d_theta):
            return 0.0

        return float(d_theta / ds)

    def add_curv_cols(df, n_cols: int=4, dist_interval: int=20):
        out = df.sort_values(["track", "year", "lap_id", "distance"]).copy()

        base_cols = ["c", "c_smooth"]
        band_cols = [f"cb{i}" for i in range(1, n_cols + 1)] + [f"ca{i}" for i in range(1, n_cols + 1)]
        for col in base_cols + band_cols:
            if col not in out.columns:
                out[col] = 0.0

        grouped_data = out.groupby(["track", "year", "lap_id"], sort=False)
        for (_, _, _), lap_df in grouped_data:
            idx = lap_df.index.to_numpy()
            distance = lap_df["distance"].to_numpy(dtype=float)

            # Smooth x/y to reduce noise:
            x_raw = lap_df["x"].to_numpy(dtype=float)
            y_raw = lap_df["y"].to_numpy(dtype=float)
            x = (pd.Series(x_raw)
                .rolling(window=7, center=True, min_periods=1).median()
                .rolling(window=15, center=True, min_periods=1).mean()
                .to_numpy(dtype=float))
            y = (pd.Series(y_raw)
                .rolling(window=7, center=True, min_periods=1).median()
                .rolling(window=15, center=True, min_periods=1).mean()
                .to_numpy(dtype=float))
            
            kappa = Curvature.get_curvature(x, y)
            k = np.abs(kappa)
            n = len(distance)

            ca_bands = np.zeros((n_cols, n), dtype=float)
            cb_bands = np.zeros((n_cols, n), dtype=float)

            for band in range(n_cols):
                lo_m = band * dist_interval
                hi_m = (band + 1) * dist_interval
                for i in range(n):
                    # Curvature Ahead (ca):
                    left_fwd  = np.searchsorted(distance, distance[i] + lo_m, side="left")
                    right_fwd = np.searchsorted(distance, distance[i] + hi_m, side="right")
                    if right_fwd > left_fwd:
                        ca_bands[band, i] = float(np.mean(k[left_fwd:right_fwd]))
                    # Curvature Behind (cb):
                    left_bwd = np.searchsorted(distance, distance[i] - hi_m, side="left")
                    right_bwd = np.searchsorted(distance, distance[i] - lo_m, side="right")
                    if right_bwd > left_bwd:
                        cb_bands[band, i] = float(np.mean(k[left_bwd:right_bwd]))

            weight_c    = 0.40
            weight_band = 0.60 / n_cols
            base = weight_c * k
            for band in range(n_cols):
                base += weight_band * ca_bands[band]

            c_smooth = (pd.Series(base)
                        .rolling(window=11, center=True, min_periods=1).median()
                        .rolling(window=31, center=True, min_periods=1).mean()
                        .to_numpy(dtype=float))
            
            # Get signed smoothness for plotting (pos/neg vals)
            c_signed_smooth = (pd.Series(kappa)
                        .rolling(window=11, center=True, min_periods=1).median()
                        .rolling(window=31, center=True, min_periods=1).mean()
                        .to_numpy(dtype=float))

            out.loc[idx, "c"] = k
            out.loc[idx, "c_smooth"] = c_smooth # for model training 
            out.loc[idx, "c_signed_smooth"] = c_signed_smooth # for plotting only
            
            for band in range(n_cols):
                out.loc[idx, f"ca{band + 1}"] = ca_bands[band]
                out.loc[idx, f"cb{band + 1}"] = cb_bands[band]
        return out