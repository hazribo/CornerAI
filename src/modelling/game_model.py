from pathlib import Path
import numpy as np
import pandas as pd
import time
import re
# other file imports:
from track_plots import PlotTrackMaps
from model_utils import Curvature, build_centreline, project_to_centreline, build_track_ground_truth, add_should_brake, add_should_throttle, add_labels
# model imports:
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

F125_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "f1-25" / "laps"
MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "f1-25"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    df = add_labels(df, LABELS)

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

                    time_match = re.search(r"_(?:Q|R|P\d)_([\d\.]+)_", fp.stem)
                    df["laptime"] = float(time_match.group(1)) if time_match else pd.NA
                    frames.append(df)
                except Exception as e:
                    print(f"fail reading {fp}: {e}")
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["track", "difficulty", "year", "lap_id", "distance"]).reset_index(drop=True)

# TODO: change top_pct once more laps have been collected.
# Very few laps currently, so top_pct = 0.2 would remove too many laps.
def filter_fast_laps(df: pd.DataFrame, top_pct: float = 0.8) -> pd.DataFrame:
    fast_ids = set()
    for _, track_df in df.groupby("track"):
        lap_times = track_df.groupby("lap_id")["laptime"].first().dropna().sort_values()
        n_keep = max(1, int(len(lap_times) * top_pct))
        fast_ids.update(lap_times.index[:n_keep])
    return df[df["lap_id"].isin(fast_ids)]

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
        
        metrics_log = []

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

            # Get all metrics for brake and throttle predictions:
            brake_preds = brake_model.predict(X.iloc[test_idx])
            throttle_preds = throttle_model.predict(X.iloc[test_idx])
            
            b_acc = accuracy_score(y_brake.iloc[test_idx], brake_preds)
            b_prec = precision_score(y_brake.iloc[test_idx], brake_preds, average='macro', zero_division=0)
            b_rec = recall_score(y_brake.iloc[test_idx], brake_preds, average='macro', zero_division=0)
            b_f1 = f1_score(y_brake.iloc[test_idx], brake_preds, average='macro')
            t_acc = accuracy_score(y_throttle.iloc[test_idx], throttle_preds)
            t_prec = precision_score(y_throttle.iloc[test_idx], throttle_preds, average='macro', zero_division=0)
            t_rec = recall_score(y_throttle.iloc[test_idx], throttle_preds, average='macro', zero_division=0)
            t_f1 = f1_score(y_throttle.iloc[test_idx], throttle_preds, average='macro')
            
            end_time = time.perf_counter() - start_time
            print(f"[{track_name}] Brake F1 (Macro): {b_f1:.4f} ({end_time:.2f}s)")
            print(f"[{track_name}] Throttle F1 (Macro): {t_f1:.4f} ({end_time:.2f}s)")
            
            metrics_log.append({
                "Circuit": track_name,
                "Brake_Accuracy": b_acc,
                "Brake_Precision": b_prec,
                "Brake_Recall": b_rec,
                "Brake_F1": b_f1,
                "Throttle_Accuracy": t_acc,
                "Throttle_Precision": t_prec,
                "Throttle_Recall": t_rec,
                "Throttle_F1": t_f1,
            })

            bundle.models_by_track[str(track_name)] = {
                "brake": brake_model,
                "throttle": throttle_model,
            }

        metrics_df = pd.DataFrame(metrics_log)
        overall_row = pd.DataFrame([{
            "Circuit": "Overall (All Circuits)",
            "Brake_Accuracy": metrics_df["Brake_Accuracy"].mean(),
            "Brake_Precision": metrics_df["Brake_Precision"].mean(),
            "Brake_Recall": metrics_df["Brake_Recall"].mean(),
            "Brake_F1": metrics_df["Brake_F1"].mean(),
            "Throttle_Accuracy": metrics_df["Throttle_Accuracy"].mean(),
            "Throttle_Precision": metrics_df["Throttle_Precision"].mean(),
            "Throttle_Recall": metrics_df["Throttle_Recall"].mean(),
            "Throttle_F1": metrics_df["Throttle_F1"].mean(),
        }])
        metrics_df = pd.concat([overall_row, metrics_df], ignore_index=True)
        
        csv_path = MODEL_OUTPUT_DIR / "game_lap_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"\nClassification metrics saved to {csv_path}.")

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
    
    for track_name, track_df in fast_laps.groupby("track", sort=False):
        cl = build_centreline(fast_laps, track=str(track_name), bin_m=5.0)
        gt = build_track_ground_truth(fast_laps, track=str(track_name), cl=cl, bin_m=5.0)
        cl_by_track[str(track_name)] = cl
        gt_by_track[str(track_name)] = gt

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

    ### Get global constellation map:
    all_laps = model.predict_probability(fast_laps.copy())
    all_laps = all_laps[~all_laps["track"].str.contains("monaco", case=False, na=False)] # remove monaco - outlier
    #all_laps = all_laps[~all_laps["track"].str.contains("shanghai", case=False, na=False)]
    #all_laps = all_laps[~all_laps["track"].str.contains("montreal", case=False, na=False)]
    #all_laps = all_laps[~all_laps["track"].str.contains("suzuka", case=False, na=False)]
    PlotTrackMaps.plot_global_state_constellation(
        laps=all_laps,
        out_dir=MODEL_OUTPUT_DIR,
        speed_col="speed",
        curv_col="c_smooth",
    )
    print(f"Saved plots/graphs to {MODEL_OUTPUT_DIR}.")