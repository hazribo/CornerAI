from pathlib import Path
import numpy as np
import pandas as pd
import time
# plot imports:
from track_plots import PlotTrackMaps
# model imports:
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import joblib

HISTORICAL_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "historical"
MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "historical"
# Cache for processed historical data:
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "historical"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# for circuits with multiple event names -> set to same track name:
TRACK_ALIASES = {
    "Styrian_Grand_Prix": "Austrian_Grand_Prix",
    "Brazilian_Grand_Prix": "São_paulo_Grand_Prix",
    "70th_Anniversary_Grand_Prix": "British_Grand_Prix",
    "Mexican_Grand_Prix": "Mexico_City_Grand_Prix",
}

# feature columns and label details for RF model:
FEATURE_COLS = ["time", "distance", "x", "y", "z", "speed", "throttle", "brake", "rpm", "gear", "drs",
                "c", "cb1", "ca1", "c_smooth"] # curvature, curvature before 100m, curvature after 100m + smoothed curvature
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

    df = load_historical_laps()
    df = add_curvature_features(df)
    df = add_labels(df)

    df.to_pickle(cache_path)
    return df

def load_historical_laps():
    frames = []
    if HISTORICAL_PROCESSED_DIR.exists():
        for track_dir in [p for p in HISTORICAL_PROCESSED_DIR.iterdir() if p.is_dir()]:
            for fp in track_dir.rglob("*.csv"):
                try:
                    year = int("".join(ch for ch in fp.stem[:4] if ch.isdigit()))
                    df = pd.read_csv(fp)
                    df = df.drop(columns=["source"], errors="ignore")
                    # Get track name - either from directory, or its alias in TRACK_ALIASES:
                    df["track"] = TRACK_ALIASES.get(track_dir.name, track_dir.name)
                    df["year"] = year
                    df["lap_id"] = fp.stem
                    # get laptime from file name, i.e. after the second underscore:
                    parts = fp.stem.split("_")
                    df["laptime"] = pd.to_numeric(parts[2], errors="coerce")
                    frames.append(df)
                except Exception as e:
                    print(f"fail reading {fp}: {e}")
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["track", "year", "lap_id", "distance"]).reset_index(drop=True)
    return out

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

def extract_events(lap, prob_col):
    # Gets a list of distances where "events" happen.
    # i.e.: upwards threshold crossing of prob_col (spaced by min separation)
    threshold = float(0.6)
    min_sep = float(40.0)

    df = lap.sort_values("distance")
    distance = df["distance"].to_numpy()
    prob = df[prob_col].to_numpy()

    idx = np.where((prob[1:] >= threshold) & (prob[:-1] < threshold))[0] + 1

    events: list[float] = []
    in_event = False

    for i in range(len(prob)):
        p = float(prob[i])

        if not in_event:
            prev_p = float(prob[i - 1]) if i > 0 else 0.0
            if p >= threshold and prev_p <= (1 - threshold):
                events.append(float(distance[i]))
                in_event = True
        else:
            if p <= (1 - threshold):
                in_event = False
    
    return events

def nearest_event(lap: float, refs: list[float]):
    tolerance = float(80.0)
    if not refs:
        return None
    array = np.asarray(refs)
    i = int(np.argmin(np.abs(array - float(lap))))
    best = float(array[i])
    return best if abs(best - float(lap)) <= tolerance else None

def build_references(scored_laps: pd.DataFrame, track: str, mode: str):
    top_percent = float(0.2)

    if mode not in ["brake", "throttle"]:
        raise ValueError(f"{mode} not 'brake' or 'throttle'.")

    # get probability for correct attribute (brake or throttle)
    prob_col = "p_brake_zone" if mode == "brake" else "p_throttle_zone"

    df = scored_laps.loc[scored_laps["track"].astype(str) == str(track)].copy()
    lap_times = (df.groupby("lap_id")["laptime"].first().sort_values())

    n_keep = max(1, int(round(len(lap_times) * float(top_percent))))
    keep_ids = set(lap_times.index[:n_keep])

    # get event lists (one per lap):
    event_list: list[list[float]] = []
    df_filter = df.loc[df["lap_id"].isin(keep_ids)]
    for _, lap_df in df_filter.groupby("lap_id"):
        event_list.append(extract_events(lap_df, prob_col))
    
    # get median by event index:
    # (this is to stop "pedantic" improvements like "brake 1 metre earlier")
    max_len = max((len(ev) for ev in event_list), default=0)
    true_events: list[float] = []

    for i in range(max_len):
        values = [event[i] for event in event_list if len(event) > i]
        if values:
            true_events.append(float(np.median(values)))

    return true_events

def advice(lap: pd.DataFrame, ref_brake: list[float], ref_throttle: list[float]):
    df = lap.sort_values("distance")
    brake = extract_events(df, "p_brake_zone")
    throttle = extract_events(df, "p_throttle_zone")
    rows: list[dict] = []

    def add(mode: str, events: list[float], refs: list[float]):
        remaining = list(events)

        for i, ref_d in enumerate(refs):
            lap_d = nearest_event(ref_d, remaining)
            if lap_d is None:
                continue
            remaining.remove(lap_d)

            delta = float(lap_d - ref_d)
            # "placeholder" advice for now:
            if mode == "brake":
                advice = f"Brake {delta:.1f}m earlier" if delta > 0 else f"Brake {delta*-1:.1f}m later"
            else:
                advice = f"Throttle {delta:.1f}m earlier" if delta > 0 else f"Throttle {delta*-1:.1f}m later"

            rows.append(
                {
                    "mode": mode,
                    "corner_index": i,
                    "lap_distance": lap_d,
                    "ref_distance": ref_d,
                    "delta": delta,
                    "advice": advice
                }
            )
    add("brake", brake, ref_brake)
    add("throttle", throttle, ref_throttle)
    return pd.DataFrame(rows).sort_values(["mode", "corner_index"]).reset_index()

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
        path = output / "lap_model.joblib"
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
    model_path = MODEL_OUTPUT_DIR / "lap_model.joblib"
    if not model_path.exists():
        model = RandomForestModel.train_models(df)
        path = model.save_model()
        print(f"Saved model to {path}.")
    else:
        model = RandomForestModel.load_model(model_path)
        print(f"Loaded model from {model_path}.")

    scored = model.predict_probability(df)

    plot_paths = PlotTrackMaps.plot_braking_zones_by_track(
        scored,
        out_dir=MODEL_OUTPUT_DIR,
        prob_threshold=0.5,
        zone_col="p_brake_zone",
    )
    dist_paths = PlotTrackMaps.plot_curvature_over_distance(
        scored,
        track="Dutch_Grand_Prix",
        out_dir=MODEL_OUTPUT_DIR,
        lap_id=None,
    )
    print(f"saved {len(plot_paths)} plots and 1 graph to {MODEL_OUTPUT_DIR}")