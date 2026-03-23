from pathlib import Path
import numpy as np
import pandas as pd
import time
import re
# other file imports:
from track_plots import PlotTrackMaps
from game_advice import build_references_from_gt, advice, write_advice
# model imports:
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy.spatial import cKDTree # For nearest-neighbour; centreline

F125_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "f1-25" / "laps"
MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "f1-25"
# Cache for processed historical data:
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "f1-25"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# feature columns and label details for RF model:
N_COLS_DEFAULT = 4
FEATURE_COLS = ["distance", "x", "y", "z",
                "c", "c_smooth", "difficulty", # curvature, curvature before 100m, curvature after 100m + smoothed curvature
                *[f"ca{i}" for i in range(1, N_COLS_DEFAULT + 1)], 
                ]

LABELS = {
    "brake_threshold": 0.1,
    "brake_lift_min": 0.05,
    "throttle_threshold": 0.1,
    "throttle_lift_min": 0.05,
    "brake_window_min": 5.0, # buffer to brake zone
}

def load_build_cache(
    cache_name: str = "laps_cached.pkl",
    rebuild: bool = False,
    cache_dir=CACHE_DIR,
) -> "pd.DataFrame":
    cache_path = cache_dir / cache_name
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path)

    df = load_game_laps()
    df = Curvature.add_curv_cols(df, n_cols=N_COLS_DEFAULT, dist_interval=50)
    df = add_labels(df)

    df.to_pickle(cache_path)
    return df

def load_game_laps():
    frames = []
    laps_root = F125_PROCESSED_DIR
    if not laps_root.exists():
        return pd.DataFrame()

    for track_dir in [p for p in laps_root.iterdir() if p.is_dir()]:
        for difficulty_dir in [p for p in track_dir.iterdir() if p.is_dir()]:
            for fp in difficulty_dir.rglob("*.csv"):
                try:
                    m = re.search(r"(19|20)\d{2}", fp.stem)
                    year = int(m.group(0)) if m else None

                    df = pd.read_csv(fp).drop(columns=["source"], errors="ignore")
                    df["track"] = track_dir.name
                    df["difficulty"] = difficulty_dir.name
                    df["year"] = year
                    df["lap_id"] = fp.stem

                    parts = fp.stem.split("_")
                    df["laptime"] = pd.to_numeric(parts[2], errors="coerce") if len(parts) > 2 else pd.NA
                    frames.append(df)
                except Exception as e:
                    print(f"fail reading {fp}: {e}")
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["track", "difficulty", "year", "lap_id", "distance"]).reset_index(drop=True)

def label_window_distance(distance_m, event_idx, window_min):
    event_distance_m = distance_m[event_idx]
    return (np.abs(distance_m  - event_distance_m) <= window_min).astype(np.int32)

def add_labels(df: pd.DataFrame):
    out = df.sort_values(["track", "year", "lap_id", "distance"]).copy()
    out["y_brake_zone"] = 0
    out["y_throttle_zone"] = 0

    brake_on = LABELS["brake_threshold"]
    throttle_on = LABELS["throttle_threshold"]
    brake_off = LABELS["brake_lift_min"]
    throttle_off = LABELS["throttle_lift_min"]
    brake_window_min = LABELS["brake_window_min"]

    grouped_data = out.groupby(["track", "year", "lap_id"], sort=False)

    for (_, _, _), lap_df in grouped_data:
        idx = lap_df.index.to_numpy()

        distance = lap_df["distance"].to_numpy()
        brake = lap_df["brake"].to_numpy()
        throttle = lap_df["throttle"].to_numpy()

        # calc both brake and throttle deltas:
        brake_delta = np.diff(brake, prepend=brake[0])

        # get braking point and throttle zones:
        brake_start = (
            brake >= brake_on) & (
            brake_delta >= brake_off) & (
            throttle <= throttle_off
        )
        brake_zone = np.zeros(len(lap_df), dtype=np.int32)
        for event_idx in np.flatnonzero(brake_start):
            brake_zone = np.maximum(brake_zone, label_window_distance(distance, event_idx, brake_window_min))

        throttle_zone = ((throttle >= throttle_on) & (brake <= brake_off)).astype(np.int32)

        out.loc[idx, "y_brake_zone"] = brake_zone
        out.loc[idx, "y_throttle_zone"] = throttle_zone

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

# TODO: change top_pct once more laps have been collected.
# Very few laps currently, so top_pct = 0.2 would remove too many laps.
def filter_fast_laps(df: pd.DataFrame, top_pct: float = 0.7) -> pd.DataFrame:
    fast_ids = set()
    for _, track_df in df.groupby("track"):
        lap_times = track_df.groupby("lap_id")["laptime"].first().dropna().sort_values()
        n_keep = max(1, int(len(lap_times) * top_pct))
        fast_ids.update(lap_times.index[:n_keep])
    return df[df["lap_id"].isin(fast_ids)]

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

def add_should_brake(
    lap_df: pd.DataFrame,
    gt: pd.DataFrame,
    speed_margin: float = 0.03,
    brake_prob_min: float = 0.5,
) -> pd.DataFrame:
    
    if lap_df.empty or gt.empty:
        out = lap_df.copy()
        out["should_brake"] = 0
        return out

    out = pd.merge_asof(
        lap_df.sort_values("cl_dist"),
        gt[["cl_dist", "x_exp", "y_exp", "c_exp", "c_exp_ahead", "norm_speed_exp", "speed_exp", "p_brake_exp"]].sort_values("cl_dist"),
        on="cl_dist",   # Use centreline distance instead of just distance
        direction="nearest",
    )

    out["should_brake"] = (
        (out["norm_speed"] > out["norm_speed_exp"] + float(speed_margin)) &
        (out["p_brake_zone"] >= float(brake_prob_min))
    ).astype(int)

    return out

def add_should_throttle(
    lap_df: pd.DataFrame,
    gt: pd.DataFrame,
    throttle_prob_min: float = 0.5,
) -> pd.DataFrame:
    
    if lap_df.empty or gt.empty:
        out = lap_df.copy()
        out["should_throttle"] = 0
        return out

    merge_cols = ["cl_dist"]
    out = pd.merge_asof(
        lap_df.sort_values("cl_dist"),
        gt[merge_cols].sort_values("cl_dist"),
        on="cl_dist",
        direction="nearest",
    )

    out["should_throttle"] = (out["p_throttle_zone"] >= float(throttle_prob_min)).astype(int)
    return out

class RandomForestModel:
    def __init__(self):
        self.models_by_track: dict[str, dict[str, RandomForestClassifier]] = {}
        self.feature_cols: list[str]

    @staticmethod
    def rf_model(seed: int = 19):
        return RandomForestClassifier(
            n_estimators = 100,      
            max_depth = 18,        
            min_samples_leaf = 5,  
            min_samples_split = 10,
            max_features = "sqrt",   
            bootstrap = True,
            max_samples = 0.8,     
            n_jobs = -1,
            class_weight = "balanced_subsample",
            random_state = seed,
            verbose = 0,
        )
    
    def save_model(self, output: Path = MODEL_OUTPUT_DIR):
        output.mkdir(parents=True, exist_ok=True)
        path = output / "game_model.joblib"
        joblib.dump(
            {
                "models_by_track": self.models_by_track,
                "feature_cols": self.feature_cols
            },
            path,
        )
        return path

    @staticmethod
    def load_model(path: Path):
        data = joblib.load(path)
        m = RandomForestModel()
        m.models_by_track = data["models_by_track"]
        m.feature_cols = list(data["feature_cols"])
        return m
    
    @staticmethod
    def train_models(laps: pd.DataFrame):
        training_df = laps.copy()
        bundle = RandomForestModel()
        bundle.feature_cols = list(FEATURE_COLS)

        for track_name, track_df in training_df.groupby("track", sort=False):
            start_time = time.perf_counter()
            X = track_df[FEATURE_COLS]
            y_brake = track_df["y_brake_zone"]
            y_throttle = track_df["y_throttle_zone"]
            groups = track_df["lap_id"]

            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=19)
            train_idx, test_idx = next(splitter.split(X, y_brake, groups=groups))

            brake_model = RandomForestModel.rf_model(seed=19)
            throttle_model = RandomForestModel.rf_model(seed=23)

            brake_model.fit(X.iloc[train_idx], y_brake.iloc[train_idx])
            throttle_model.fit(X.iloc[train_idx], y_throttle.iloc[train_idx])

            brake_acc = (brake_model.predict(X.iloc[test_idx]) == y_brake.iloc[test_idx]).mean()
            throttle_acc = (throttle_model.predict(X.iloc[test_idx]) == y_throttle.iloc[test_idx]).mean()
            end_time = time.perf_counter() - start_time
            print(f"[{track_name}] Accuracies: brake {brake_acc:.4f}, throttle {throttle_acc:.4f}")
            print(f"Time elapsed for {track_name}: {end_time:.2f}s")

            bundle.models_by_track[str(track_name)] = {
                "brake": brake_model,
                "throttle": throttle_model,
            }

        return bundle
    
    def predict_probability(self, laps: pd.DataFrame):
        laps["p_brake_zone"] = np.nan
        laps["p_throttle_zone"] = np.nan

        for track_name, track_df in laps.groupby("track", sort=False):
            track_name = str(track_name)
            if track_name not in self.models_by_track:
                raise ValueError(f"No trained model found for track='{track_name}'.")

            X = track_df[self.feature_cols]
            brake_model = self.models_by_track[track_name]["brake"]
            throttle_model = self.models_by_track[track_name]["throttle"]
            laps.loc[track_df.index, "p_brake_zone"] = brake_model.predict_proba(X)[:, 1]
            laps.loc[track_df.index, "p_throttle_zone"] = throttle_model.predict_proba(X)[:, 1]

        return laps

if __name__ == "__main__":
    df = load_build_cache()
    print(f"cache loaded: rows={len(df):,} cols={df.shape[1]:,}")
    # Separate top 20% of laps for training:
    fast_laps = filter_fast_laps(df, top_pct=0.2)
    # Generate model if doesn't already exist; otherwise, use existing model:
    model_path = MODEL_OUTPUT_DIR / "game_model.joblib"
    if not model_path.exists():
        model = RandomForestModel.train_models(fast_laps)
        path = model.save_model()
        print(f"Saved model to {path}.")
    else:
        model = RandomForestModel.load_model(model_path)
        print(f"Loaded model from {model_path}.")

    cl_by_track: dict[str, pd.DataFrame] = {}
    gt_by_track: dict[str, pd.DataFrame] = {}
    
    # Notice we loop over fast_laps, NOT scored!
    for track_name, track_df in fast_laps.groupby("track", sort=False):
        cl = build_centreline(fast_laps, track=str(track_name), bin_m=5.0)
        gt = build_track_ground_truth(fast_laps, track=str(track_name), cl=cl, bin_m=5.0)
        cl_by_track[str(track_name)] = cl
        gt_by_track[str(track_name)] = gt

    # TODO: have advice running real-time in f1_25_listener.py rather than here after training.
    # TESTING ADVICE:
    target_track = "1 melbourne"
    target_lap_id = Path(r"Z:\CornerAI\data\processed\f1-25\laps\1 melbourne\lap_1.csv")

    player_lap = pd.read_csv(target_lap_id)
    player_lap["track"] = target_track
    player_lap["year"] = 0
    player_lap["lap_id"] = "player"
    player_lap["difficulty"] = 0

    player_lap = Curvature.add_curv_cols(player_lap, n_cols=N_COLS_DEFAULT, dist_interval=50)
    player_lap = model.predict_probability(player_lap)

    cl = cl_by_track.get(target_track)        
    player_lap = project_to_centreline(player_lap, cl)  
    gt = gt_by_track.get(target_track, pd.DataFrame())
    lap_df = add_should_brake(player_lap, gt)
    lap_df = add_should_throttle(lap_df, gt).sort_values("cl_dist")  

    track_name = target_track

    ref_brake = build_references_from_gt(gt, mode="brake")
    ref_throttle = build_references_from_gt(gt, mode="throttle")
    advice_df = advice(lap_df, ref_brake, ref_throttle, gt=gt)

    txt_path = MODEL_OUTPUT_DIR / f"{track_name}_player_advice.txt"
    write_advice(advice_df, txt_path, track_name, lap_id="player")
    print(f"Saved advice: {txt_path}.")
        ##############################################################
    # Save to csv per track:
    for t, gt in gt_by_track.items():
        gt.to_csv(MODEL_OUTPUT_DIR / f"{t}_ground_truth.csv")

        cl = cl_by_track[t]
        current_track_laps = fast_laps[fast_laps["track"] == t].copy()
        current_track_laps = project_to_centreline(current_track_laps, cl)
        current_track_laps = model.predict_probability(current_track_laps) 

        PlotTrackMaps.plot_track_dashboard(
            current_track_laps,
            track_name=t,
            out_dir=MODEL_OUTPUT_DIR,
            curv_col="c_signed_smooth" # used signed curvature for plots
        )
    print(f"Saved plots/graphs to {MODEL_OUTPUT_DIR}.")