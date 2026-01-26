from pathlib import Path
import numpy as np
import pandas as pd
import time
# plotly imports:
import plotly.graph_objects as go
import plotly.io as pio
# model imports:
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import joblib

HISTORICAL_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "historical"
MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

# for circuits with multiple event names -> set to same track name:
TRACK_ALIASES = {
    "Styrian_Grand_Prix": "Austrian_Grand_Prix",
    "Brazilian_Grand_Prix": "São_paulo_Grand_Prix",
    "70th_Anniversary_Grand_Prix": "British_Grand_Prix",
    "Mexican_Grand_Prix": "Mexico_City_Grand_Prix",
}

# feature columns and label details for RF model:
FEATURE_COLS = ["time", "distance", "x", "y", "z", "speed", "throttle", "brake", "rpm", "gear", "drs"]
LABELS = {
    "brake_threshold": 0.1,
    "brake_lift_min": 0.05,
    "throttle_lift_min": 0.1,
    "throttle_threshold": 0.2,
    "brake_window_min": 10.0,
    "throttle_window_min": 10.0
}

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

        distance = lap_df["distance"].to_numpy(dtype=float)
        brake = lap_df["brake"].to_numpy(dtype=float)
        throttle = lap_df["throttle"].to_numpy(dtype=float)

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

class RandomForestModel:
    def __init__(self):
        self.models_by_track: dict[str, dict[str, RandomForestClassifier]] = {}
        self.feature_cols: list[str]

    @staticmethod
    def rf_model(seed: int = 19):
        return RandomForestClassifier(
            n_estimators = 100,
            max_depth = 20,
            min_samples_leaf = 20,
            max_features = "sqrt",
            max_samples = 0.6,
            n_jobs = -1,
            class_weight = "balanced_subsample",
            random_state = seed,
            verbose=1 # get some output to track model status
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
            print(f"Time elapsed for {track_name}: {end_time}")

            bundle.models_by_track[str(track_name)] = {
                "brake": brake_model,
                "throttle": throttle_model,
            }

        return bundle
    
    def predict_probability(self, laps: pd.DataFrame):
        scored = laps.copy()
        scored["p_brake_zone"] = np.nan
        scored["p_throttle_zone"] = np.nan

        for track_name, track_df in scored.groupby("track", sort=False):
            track_name = str(track_name)
            if track_name not in self.models_by_track:
                raise ValueError(f"No trained model found for track='{track_name}'.")

            X = track_df[self.feature_cols]
            brake_model = self.models_by_track[track_name]["brake"]
            throttle_model = self.models_by_track[track_name]["throttle"]
            scored.loc[track_df.index, "p_brake_zone"] = brake_model.predict_proba(X)[:, 1]
            scored.loc[track_df.index, "p_throttle_zone"] = throttle_model.predict_proba(X)[:, 1]

        return scored
    
class PlotTrackMaps:
    @staticmethod
    def plot_braking_zones_by_track(
        laps: pd.DataFrame,
        out_dir: Path = MODEL_OUTPUT_DIR,
        prob_threshold: float = 0.5,
        zone_col: str = "p_brake_zone",
        max_zone_points: int = 15000,
    ) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []

        required = {"track", "lap_id", "distance", "x", "z", zone_col}
        missing = required - set(laps.columns)
        if missing:
            raise ValueError(f"Missing required columns for plotting: {sorted(missing)}")

        df = laps.copy()

        for track_name, track_df in df.groupby("track", sort=False):
            track_name = str(track_name)

            first_lap_id = track_df["lap_id"].astype(str).iloc[0]
            base_lap = track_df.loc[track_df["lap_id"].astype(str) == first_lap_id].sort_values("distance")

            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=base_lap["x"].to_numpy(dtype=float),
                    y=base_lap["z"].to_numpy(dtype=float),
                    mode="lines",
                    name=f"Track map (lap_id={first_lap_id})",
                    line=dict(color="rgba(0,0,0,0.6)", width=2),
                    hoverinfo="skip",
                )
            )

            # --- Braking zone points (raw x/z) ---
            if zone_col.startswith("p_"):
                zone_rows = track_df.loc[track_df[zone_col].astype(float) >= float(prob_threshold)].copy()
            else:
                zone_rows = track_df.loc[track_df[zone_col].astype(int) == 1].copy()

            if not zone_rows.empty:
                # keep plot responsive if there are tons of points
                if len(zone_rows) > max_zone_points:
                    zone_rows = zone_rows.sample(n=max_zone_points, random_state=19)

                zone_rows = zone_rows.sort_values("distance")

                fig.add_trace(
                    go.Scattergl(
                        x=zone_rows["x"].to_numpy(dtype=float),
                        y=zone_rows["z"].to_numpy(dtype=float),
                        mode="markers",
                        name=f"Braking zones ({zone_col})",
                        marker=dict(size=6, color="rgba(220,20,60,0.95)"),
                        customdata=np.c_[
                            zone_rows["distance"].to_numpy(dtype=float),
                            zone_rows[zone_col].to_numpy(dtype=float),
                        ],
                        hovertemplate="BRAKE<br>dist=%{customdata[0]:.1f}m<br>p=%{customdata[1]:.3f}<extra></extra>",
                    )
                )

            fig.update_layout(
                title=f"{track_name} — braking zones ({zone_col}{'' if not zone_col.startswith('p_') else f' >= {prob_threshold}'})",
                template="plotly_white",
                xaxis_title="x",
                yaxis_title="z",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            out_path = out_dir / f"{track_name}_braking_zones.html"
            pio.write_html(fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn")
            outputs.append(out_path)

        return outputs

if __name__ == "__main__":
    f1_laps = load_historical_laps()
    labelled_laps = add_labels(f1_laps)
    # debug:
    # print(f"size of f1_laps: {len(f1_laps)}")
    #model = RandomForestModel.train_models(labelled_laps)
    #path = model.save_model()
    #print(f"saved model to {path}")

    model = RandomForestModel.load_model("Z:/CornerAI/data/models/lap_model.joblib")

    scored = model.predict_probability(labelled_laps)
    plot_paths = PlotTrackMaps.plot_braking_zones_by_track(
        scored,
        out_dir=MODEL_OUTPUT_DIR,
        prob_threshold=0.5,
        zone_col="p_brake_zone",
    )
    print(f"saved {len(plot_paths)} plots to {MODEL_OUTPUT_DIR}")