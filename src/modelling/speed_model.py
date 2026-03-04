from pathlib import Path
import numpy as np
import pandas as pd
import time
import re
# other file imports:
from track_plots import PlotTrackMaps
from game_model import load_build_cache, RandomForestModel
# model imports:
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import joblib

# split into train and test using different tracks,
# fit model to predict speed, based on past curvature, current curvature, and oncoming curvature

MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "f1-25"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "f1-25-speed" 
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    df = load_build_cache(cache_dir=CACHE_DIR)
    print(f"cache loaded: rows={len(df):,} cols={df.shape[1]:,}")

    model_path = MODEL_OUTPUT_DIR / "speed_model.joblib"
    if not model_path.exists():
        model = RandomForestModel.train_models(df)
        path = model.save_model()
        print(f"Saved model to {path}.")
    else:
        model = RandomForestModel.load_model(model_path)
        print(f"Loaded model from {model_path}.")