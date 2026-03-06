from pathlib import Path
import numpy as np
import pandas as pd
import time

from game_model import load_build_cache
from track_plots import plot_predicted_speed

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "f1-25"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "f1-25"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["c", "ca1", "cb1", "c_smooth"]
TARGET_COL = "speed"

class RandomForestModel:
    def __init__(self):
        self.models_by_track_diff: dict[str, RandomForestClassifier] = {}
        self.feature_cols: list[str]

    @staticmethod
    def rf_model(seed: int = 19):
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            min_samples_leaf=50,
            min_samples_split=100,
            max_features="log2",
            bootstrap=True,
            max_samples=0.4,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=seed,
            verbose=0,
        )

    def save_model(self, output: Path = MODEL_OUTPUT_DIR):
        output.mkdir(parents=True, exist_ok=True)
        path = output / "speed_model.joblib"
        joblib.dump(
            {
                "models_by_track_diff": self.models_by_track_diff,
                "feature_cols": self.feature_cols,
            },
            path,
        )
        return path

    @staticmethod
    def load_model(path: Path):
        data = joblib.load(path)
        m = RandomForestModel()
        m.models_by_track_diff = data["models_by_track_diff"]
        m.feature_cols = list(data["feature_cols"])
        return m

    @staticmethod
    def train_models(laps: pd.DataFrame):
        training_df = laps.copy()
        bundle = RandomForestModel()
        bundle.feature_cols = list(FEATURE_COLS)

        for (track_name, difficulty), group_df in training_df.groupby(["track", "difficulty"], sort=False):
            key = f"{track_name}_{difficulty}"
            start_time = time.perf_counter()

            X = group_df[FEATURE_COLS]
            y = group_df[TARGET_COL]
            groups = group_df["lap_id"]

            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=19)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))

            speed_model = RandomForestModel.rf_model(seed=19)
            speed_model.fit(X.iloc[train_idx], y.iloc[train_idx])

            acc = (speed_model.predict(X.iloc[test_idx]) == y.iloc[test_idx]).mean()
            end_time = time.perf_counter() - start_time
            print(f"[{track_name} | diff={difficulty}] Accuracy: {acc:.4f} | Time: {end_time:.2f}s")

            bundle.models_by_track_diff[key] = speed_model

        return bundle

    def predict(self, laps: pd.DataFrame) -> pd.DataFrame:
        laps = laps.copy()
        laps["predicted_speed"] = np.nan

        for (track_name, difficulty), track_df in laps.groupby(["track", "difficulty"], sort=False):
            key = f"{track_name}_{difficulty}"
            if key not in self.models_by_track_diff:
                raise ValueError(f"No trained model found for track='{track_name}', difficulty='{difficulty}'.")

            X = track_df[self.feature_cols]
            laps.loc[track_df.index, "predicted_speed"] = self.models_by_track_diff[key].predict(X)

        return laps

if __name__ == "__main__":
    df = load_build_cache(cache_dir=CACHE_DIR)
    print(f"Cache loaded: rows={len(df):,} cols={df.shape[1]:,}")

    model_path = MODEL_OUTPUT_DIR / "speed_model.joblib"
    if not model_path.exists():
        model = RandomForestModel.train_models(df)
        path = model.save_model()
        print(f"Saved model to {path}.")
    else:
        model = RandomForestModel.load_model(model_path)
        print(f"Loaded model from {model_path}.")
