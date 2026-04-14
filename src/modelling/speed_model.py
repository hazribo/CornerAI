from pathlib import Path
import numpy as np
import pandas as pd
import time

from game_model import load_build_cache, filter_fast_laps
from track_plots import PlotTrackMaps

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import joblib

MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "f1-25"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "f1-25"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_COLS_DEFAULT = 4
FEATURE_COLS = (
    ["c_smooth"] + 
    [f"cb{i}" for i in range(1, N_COLS_DEFAULT + 1)] + 
    [f"ca{i}" for i in range(1, N_COLS_DEFAULT + 1)]
)
TARGET_COL = "speed"

class RandomForestModel:
    def __init__(self):
        self.models_by_track: dict[str, RandomForestRegressor] = {} #
        self.feature_cols: list[str]

    @staticmethod
    def rf_model(seed: int = 19):
        return RandomForestRegressor(
            n_estimators=100,         
            max_depth=20,             
            min_samples_split=10,     
            min_samples_leaf=5,       
            max_features=1.0,         
            bootstrap=True,
            max_samples=0.8,         
            n_jobs=-1,
            random_state=seed,
            verbose=0,
        )

    def save_model(self, output: Path = MODEL_OUTPUT_DIR):
        output.mkdir(parents=True, exist_ok=True)
        path = output / "speed_model.joblib"
        joblib.dump(
            {
                "models_by_track": self.models_by_track, 
                "feature_cols": self.feature_cols,
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
        maes, r2s, rmses = [], [], []

        for track_name, track_df in training_df.groupby("track", sort=False):
            start_time = time.perf_counter()

            X = track_df[FEATURE_COLS]
            y = track_df[TARGET_COL]
            groups = track_df["lap_id"]

            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=19)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))

            speed_model = RandomForestModel.rf_model(seed=19)
            speed_model.fit(X.iloc[train_idx], y.iloc[train_idx])

            preds = speed_model.predict(X.iloc[test_idx])
            mae = mean_absolute_error(y.iloc[test_idx], preds)
            r2 = r2_score(y.iloc[test_idx], preds)
            rmse = root_mean_squared_error(y.iloc[test_idx], preds)
            maes.append(mae)
            r2s.append(r2)
            rmses.append(rmse)

            
            end_time = time.perf_counter() - start_time
            print(f"[{track_name}]: MAE {mae:.4f}, R2 {r2:.4f}, RMSE {rmse:.4f}")
            print(f"Time elapsed for {track_name}: {end_time:.2f}s")

            bundle.models_by_track[str(track_name)] = speed_model 
        print(f"Average MAE: {np.mean(maes):.4f}")
        print(f"Average R2: {np.mean(r2s):.4f}")
        print(f"Average RMSE: {np.mean(rmses):.4f}")
        return bundle

    def predict(self, laps: pd.DataFrame) -> pd.DataFrame:
        laps = laps.copy()
        laps["predicted_speed"] = np.nan

        for track_name, track_df in laps.groupby("track", sort=False):
            track_name = str(track_name)
            if track_name not in self.models_by_track:
                raise ValueError(f"No trained model found for track='{track_name}'.")

            X = track_df[self.feature_cols]
            laps.loc[track_df.index, "predicted_speed"] = self.models_by_track[track_name].predict(X)

        return laps

if __name__ == "__main__":
    df = load_build_cache(cache_dir=CACHE_DIR, rebuild=False) 
    print(f"cache loaded: rows={len(df):,} cols={df.shape[1]:,}")
    fast_laps = filter_fast_laps(df, top_pct=0.7)
    
    model_path = MODEL_OUTPUT_DIR / "speed_model.joblib"
    model = RandomForestModel.train_models(fast_laps) 
    path = model.save_model()
    print(f"Saved model to {path}.")
    
    scored = model.predict(df) 
    track_names = scored["track"].unique()
    for track in track_names:
        PlotTrackMaps.plot_predicted_speed(
            laps=scored,
            track_name=track, 
            out_dir=MODEL_OUTPUT_DIR,
            speed_col="predicted_speed"
        )
    print(f"Predicted speed heatmaps saved to {MODEL_OUTPUT_DIR}.")