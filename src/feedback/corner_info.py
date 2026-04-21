import fastf1 as ff1
import warnings
import logging
from pathlib import Path
logging.getLogger("fastf1").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable cache - use same as raw historical data (non-csv):
cache_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "historical"
cache_dir.mkdir(parents=True, exist_ok=True)
ff1.Cache.enable_cache(str(cache_dir))

CORNER_CACHE = {}

class GetTurnNo():
    def __init__(self):
        self.year = 2025

    def load_corner_data(self, circuit_name: str):
        session = ff1.get_session(self.year, circuit_name, "Q")
        session.load()
        circuit_info = session.get_circuit_info()
        corners = circuit_info.corners

        return corners
    
def get_corner_no(track_name: str):
    name = track_name.split(" ", 1)[-1].strip()
    if name in CORNER_CACHE:
        return CORNER_CACHE[name]
            
    try:
        #print(f"Fetching FastF1 corner data for {name}...")
        corners = GetTurnNo().load_corner_data(name)
        CORNER_CACHE[name] = corners
        
        return corners
    except Exception as e:
        print(f"Failed to get corners: {e}")
        CORNER_CACHE[name] = None
        
        return None