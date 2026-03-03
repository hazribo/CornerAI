from pathlib import Path
import numpy as np
import pandas as pd
import time
import re
# other file imports:
from track_plots import PlotTrackMaps
from game_advice import build_references, advice, write_advice
# model imports:
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import joblib

F125_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "f1-25" / "laps"
MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "f1-25"
# Cache for processed historical data:
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "f1-25"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# feature columns and label details for RF model:
FEATURE_COLS = ["time", "distance", "x", "y", "z", "speed", "throttle", "brake", "rpm", "gear", "drs",
                "c", "cb1", "ca1", "c_smooth", "difficulty"] # curvature, curvature before 100m, curvature after 100m + smoothed curvature
LABELS = {
    "brake_threshold": 0.1,
    "brake_lift_min": 0.05,
    "throttle_lift_min": 0.1,
    "throttle_threshold": 0.2,
    "brake_window_min": 10.0,
    "throttle_window_min": 10.0
}

def load_build_cache(
    cache_name: str = "laps_cached.pkl",
    rebuild: bool = False,
) -> "pd.DataFrame":
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path)

    df = load_game_laps()
    df = add_curvature_features(df)
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
    throttle_window_min = LABELS["throttle_window_min"]

    grouped_data = out.groupby(["track", "year", "lap_id"], sort=False)

    for (_, _, _), lap_df in grouped_data:
        idx = lap_df.index.to_numpy()

        distance = lap_df["distance"].to_numpy()
        brake = lap_df["brake"].to_numpy()
        throttle = lap_df["throttle"].to_numpy()

        # calc both brake and throttle deltas:
        brake_delta = np.diff(brake, prepend=brake[0])
        throttle_delta = np.diff(throttle, prepend=throttle[0])

        # get braking point and throttle zones:
        brake_start = (
            brake >= brake_on) & (
            brake_delta >= brake_off) & (
            throttle <= throttle_off
            )
        brake_zone = np.zeros(len(lap_df), dtype=np.int32)
        
        # get where throttle float is "low", to register when throttle is rising:
        low_throttle = np.r_[True, throttle[:-1] <= throttle_off]

        throttle_start = (
            throttle >= throttle_on) & (
            throttle_delta >= throttle_off) & (
            brake <= brake_off) & (
            low_throttle
            )
        throttle_zone = np.zeros(len(lap_df), dtype=np.int32)

        for event_idx in np.flatnonzero(brake_start):
            brake_zone = np.maximum(brake_zone, label_window_distance(distance, event_idx, brake_window_min))
        for event_idx in np.flatnonzero(throttle_start):
            throttle_zone = np.maximum(throttle_zone, label_window_distance(distance, event_idx, throttle_window_min))

        out.loc[idx, "y_brake_zone"] = brake_zone
        out.loc[idx, "y_throttle_zone"] = throttle_zone

    return out

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
        kappa[i] = calc_curvature(A, B, C)

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

def curvature_context(distance_m, kappa, window_m=100.0):
    d = np.asarray(distance_m, dtype=float)
    k = np.abs(np.asarray(kappa, dtype=float))

    n = len(d)
    c = k.copy()
    cb1 = np.zeros(n, dtype=float)
    ca1 = np.zeros(n, dtype=float)

    for i in range(n):
        left = np.searchsorted(d, d[i] - float(window_m), side="left")
        right = np.searchsorted(d, d[i] + float(window_m), side="right")

        if i > left:
            cb1[i] = float(np.mean(k[left:i]))
        if right > i + 1:
            ca1[i] = float(np.mean(k[i + 1:right]))

    return c, cb1, ca1

def add_curvature_features(df):
    out = df.sort_values(["track", "year", "lap_id", "distance"]).copy()
    for col in ["c", "cb1", "ca1", "c_smooth"]:
        if col not in out.columns:
            out[col] = 0.0

    grouped_data = out.groupby(["track", "year", "lap_id"], sort=False)
    for (_, _, _), lap_df in grouped_data:
        idx = lap_df.index.to_numpy()

        distance = lap_df["distance"].to_numpy(dtype=float)

        # Smooth x/y to reduce noise:
        x_raw = lap_df["x"].to_numpy(dtype=float)
        y_raw = lap_df["y"].to_numpy(dtype=float)
        x = pd.Series(x_raw).rolling(window=7, center=True, min_periods=1).median().rolling(window=15, center=True, min_periods=1).mean().to_numpy(dtype=float)
        y = pd.Series(y_raw).rolling(window=7, center=True, min_periods=1).median().rolling(window=15, center=True, min_periods=1).mean().to_numpy(dtype=float)

        kappa = get_curvature(x, y)
        c, cb1, ca1 = curvature_context(distance, kappa, window_m=100.0)

        base = (0.50 * c) + (0.25 * cb1) + (0.25 * ca1)
        c_smooth = pd.Series(base).rolling(window=11, center=True, min_periods=1).median().rolling(window=31, center=True, min_periods=1).mean().to_numpy(dtype=float)

        out.loc[idx, "c"] = c
        out.loc[idx, "cb1"] = cb1
        out.loc[idx, "ca1"] = ca1
        out.loc[idx, "c_smooth"] = c_smooth

    return out

def build_track_ground_truth(
    laps: pd.DataFrame,
    track: str,
    bin_m: float = 5.0,
) -> pd.DataFrame:
    """
    Distance-domain template for one track:
    expected x/y, curvature, speed, brake likelihood.
    """
    d = laps.loc[laps["track"].astype(str) == str(track)].copy()
    if d.empty:
        return pd.DataFrame()

    brake_col = "p_brake_zone" if "p_brake_zone" in d.columns else "y_brake_zone"
    d["dist_bin"] = (d["distance"].astype(float) / float(bin_m)).round().astype(int) * float(bin_m)

    gt = (
        d.groupby("dist_bin", as_index=False)
         .agg(
             distance=("dist_bin", "first"),
             x_exp=("x", "mean"),
             y_exp=("y", "mean"),
             c_exp=("c_smooth", "mean"),
             speed_exp=("speed", "mean"),
             p_brake_exp=(brake_col, "mean"),
         )
         .sort_values("distance")
         .reset_index(drop=True)
    )
    gt["c_exp_ahead"] = gt["c_exp"].shift(-3).fillna(gt["c_exp"])
    return gt

def add_should_brake(
    lap_df: pd.DataFrame,
    gt: pd.DataFrame,
    speed_margin: float = 0.03,
    curv_margin: float = 0.0002,
    brake_prob_min: float = 0.5,
) -> pd.DataFrame:
    
    if lap_df.empty or gt.empty:
        out = lap_df.copy()
        out["should_brake"] = 0
        return out

    out = pd.merge_asof(
        lap_df.sort_values("distance"),
        gt[["distance", "x_exp", "y_exp", "c_exp", "c_exp_ahead", "speed_exp", "p_brake_exp"]].sort_values("distance"),
        on="distance",
        direction="nearest",
    )

    out["should_brake"] = (
        (out["speed"] > out["speed_exp"] + float(speed_margin)) &
        (out["c_exp_ahead"] > out["c_exp"] + float(curv_margin)) &
        (out["p_brake_exp"] >= float(brake_prob_min))
    ).astype(int)

    return out

class RandomForestModel:
    def __init__(self):
        self.models_by_track: dict[str, dict[str, RandomForestClassifier]] = {}
        self.feature_cols: list[str]

    @staticmethod
    def rf_model(seed: int = 19):
        return RandomForestClassifier(
            n_estimators = 50,
            max_depth = 12,
            min_samples_leaf = 50,
            min_samples_split = 100,
            max_features = "log2",
            bootstrap = True,
            max_samples = 0.4,
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
    
    # Generate model if doesn't already exist; otherwise, use existing model:
    model_path = MODEL_OUTPUT_DIR / "game_model.joblib"
    if not model_path.exists():
        model = RandomForestModel.train_models(df)
        path = model.save_model()
        print(f"Saved model to {path}.")
    else:
        model = RandomForestModel.load_model(model_path)
        print(f"Loaded model from {model_path}.")

    # Score tracks and then get ground truths & advice:
    scored = model.predict_probability(df)
    gt_by_track: dict[str, pd.DataFrame] = {}
    annotated_laps: list[pd.DataFrame] = []

    for track_name, track_df in scored.groupby("track", sort=False):
        gt = build_track_ground_truth(scored, track=str(track_name), bin_m=5.0)
        gt_by_track[str(track_name)] = gt
        for _, lap_df in track_df.groupby("lap_id", sort=False):
            annotated_laps.append(add_should_brake(lap_df, gt))

    scored = pd.concat(annotated_laps, ignore_index=True)

    ##############################################################
    # testing advice:
    target_track = "1 melbourne"
    target_lap_id = Path(r"Z:\CornerAI\data\processed\f1-25\laps\1 melbourne\lap_1.csv")

    player_lap = pd.read_csv(target_lap_id)

    player_lap = add_curvature_features(player_lap)
    player_lap = model.predict_probability(player_lap)
    gt = gt_by_track.get(target_track, pd.DataFrame())
    lap_df = add_should_brake(player_lap, gt).sort_values("distance")

    track_name = target_track

    # Build references from the dataset (scored), but generate advice for the player lap (lap_df)
    ref_brake = build_references(scored, track=track_name, mode="brake")
    ref_throttle = build_references(scored, track=track_name, mode="throttle")
    advice_df = advice(lap_df, ref_brake, ref_throttle)

    txt_path = MODEL_OUTPUT_DIR / f"{track_name}_player_advice.txt"
    write_advice(advice_df, txt_path, track_name, lap_id="player")
    print(f"saved advice: {txt_path}")
    ##############################################################

    # Save to csv per track:
    for t, gt in gt_by_track.items():
        gt.to_csv(MODEL_OUTPUT_DIR / f"{t}_ground_truth.csv")

    plot_paths = PlotTrackMaps.plot_braking_zones_by_track(
        scored,
        out_dir=MODEL_OUTPUT_DIR,
        prob_threshold=0.7,
        zone_col="p_brake_zone",
    )
    track_name = str(scored["track"].astype(str).iloc[0])
    dist_paths = PlotTrackMaps.plot_curvature_over_distance(
        scored,
        track=track_name,
        out_dir=MODEL_OUTPUT_DIR,
        lap_id=None,
    )
    print(f"saved {len(plot_paths)} plots and 1 graph to {MODEL_OUTPUT_DIR}")