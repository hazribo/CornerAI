from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

class PlotTrackMaps:
    CURVATURE_COL = "c_smooth"

    @staticmethod
    def _pick_base_lap(track_df: pd.DataFrame) -> pd.DataFrame:
        first_lap_id = str(track_df["lap_id"].astype(str).iloc[0])
        return (
            track_df.loc[track_df["lap_id"].astype(str) == first_lap_id]
            .sort_values("distance")
            .reset_index(drop=True)
        )

    @staticmethod
    def _add_track_line(fig, base_lap: pd.DataFrame, lap_id: str) -> None:
        fig.add_trace(
            go.Scattergl(
                x=base_lap["x"].to_numpy(),
                y=base_lap["y"].to_numpy(),
                mode="lines",
                name=f"Track (lap_id={lap_id})",
                line=dict(color="rgba(0,0,0,0.55)", width=2),
                hoverinfo="skip",
            )
        )

    @staticmethod
    def _add_curvature_layer(fig, base_lap: pd.DataFrame) -> None:
        curv_col = PlotTrackMaps.CURVATURE_COL if PlotTrackMaps.CURVATURE_COL in base_lap.columns else "c"
        if curv_col not in base_lap.columns:
            raise ValueError("Missing curvature columns. Run add_curvature_features() first.")

        fig.add_trace(
            go.Scattergl(
                x=base_lap["x"].to_numpy(),
                y=base_lap["y"].to_numpy(),
                mode="markers",
                name=f"Curvature ({curv_col})",
                marker=dict(
                    size=4,
                    color=base_lap[curv_col].to_numpy(dtype=float),
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(title=curv_col),
                    opacity=0.85,
                ),
                hoverinfo="skip",
            )
        )

    @staticmethod
    def _select_zone_rows(
        track_df: pd.DataFrame,
        zone_col: str,
        prob_threshold: float,
        max_points: int,
        random_state: int = 19,
    ) -> pd.DataFrame:
        if zone_col not in track_df.columns:
            return track_df.iloc[0:0].copy()

        zone = track_df[zone_col]
        is_prob = np.issubdtype(zone.dtype, np.floating)

        rows = (
            track_df.loc[zone.astype(float) >= float(prob_threshold)].copy()
            if is_prob
            else track_df.loc[zone.astype(int) == 1].copy()
        )

        if rows.empty:
            return rows

        if len(rows) > int(max_points):
            rows = rows.sample(n=int(max_points), random_state=int(random_state))

        return rows.sort_values("distance")

    @staticmethod
    def _add_zone_layer(fig, zone_rows: pd.DataFrame, zone_col: str, prob_threshold: float) -> None:
        if zone_rows.empty:
            return

        curv_col = PlotTrackMaps.CURVATURE_COL if PlotTrackMaps.CURVATURE_COL in zone_rows.columns else "c"
        curv_vals = zone_rows[curv_col].to_numpy(dtype=float) if curv_col in zone_rows.columns else np.full(len(zone_rows), np.nan)

        fig.add_trace(
            go.Scattergl(
                x=zone_rows["x"].to_numpy(),
                y=zone_rows["y"].to_numpy(),
                mode="markers",
                name=f"Zones ({zone_col} ≥ {prob_threshold})",
                marker=dict(size=7, color="rgba(220,20,60,0.95)"),
                customdata=np.c_[
                    zone_rows["distance"].to_numpy(dtype=float),
                    zone_rows[zone_col].to_numpy(dtype=float),
                    curv_vals,
                ],
                hovertemplate=(
                    "ZONE"
                    "<br>dist=%{customdata[0]:.1f}m"
                    "<br>value=%{customdata[1]:.3f}"
                    f"<br>{curv_col}=%{{customdata[2]:.6f}}"
                    "<extra></extra>"
                ),
            )
        )
    
    @staticmethod
    def plot_curvature_over_distance(
        laps: "pd.DataFrame",
        track: str,
        out_dir: Path,
        lap_id: str | None = None,
        curvature_col: str = "c_smooth",
    ) -> Path:
        df = laps.copy()
        if track is not None:
            df = df.loc[df["track"].astype(str) == str(track)]
        if df.empty:
            raise ValueError("No rows found when filtered by track.")
        
        if lap_id is None:
            lap_id = str(df["lap_id"].astype(str).iloc[0])
        df = df.loc[df["lap_id"].astype(str) == str(lap_id)].sort_values("distance")
        if df.empty:
            raise ValueError("No rows found when filtered by lap_id.")
        
        if curvature_col not in df.columns:
            raise ValueError("Missing curvature column.")
        
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=df["distance"].to_numpy(),
                y=df[curvature_col].to_numpy(),
                mode="lines",
                name=f"|{curvature_col}|",
            )
        )
        title_track = str(track) if track is not None else str(df["track"].astype(str).iloc[0])
        fig.update_layout(
            title=f"{title_track} — {curvature_col} over distance (lap_id={lap_id})",
            template="plotly_white",
            xaxis_title="distance (m)",
            yaxis_title=f"|{curvature_col}|",
        )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{title_track}_{curvature_col}_over_distance.html"
        pio.write_html(fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn")
        return out_path
    
    @staticmethod
    def plot_predicted_speed(
        laps: pd.DataFrame,
        track_name: str,
        out_dir: Path,
        lap_id: str | None = None,
        speed_col: str = "predicted_speed",
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = laps.copy()
        if track_name is not None:
            df = df.loc[df["track"].astype(str) == str(track_name)]
        if df.empty:
            raise ValueError(f"No rows found when filtered by track='{track_name}'")
        
        if lap_id is None:
            lap_id = str(df["lap_id"].astype(str).iloc[0])
            
        df = df.loc[df["lap_id"].astype(str) == str(lap_id)].sort_values("distance")
        if df.empty:
            raise ValueError(f"No rows found when filtered by lap_id='{lap_id}'")
            
        if speed_col not in df.columns:
            if "speed" in df.columns:
                speed_col = "speed"
            else:
                raise ValueError(f"No speed column")

        fig = go.Figure()

        PlotTrackMaps._add_track_line(fig, df, lap_id=lap_id)
        fig.add_trace(
            go.Scattergl(
                x=df["x"].to_numpy(),
                y=df["y"].to_numpy(),
                mode="markers",
                name=f"Speed ({speed_col})",
                marker=dict(
                    size=4,
                    color=df[speed_col].to_numpy(dtype=float),
                    colorscale="Turbo",  # Blue/Green for slow, Red for fast
                    showscale=True,
                    colorbar=dict(title=f"{speed_col} (km/h)"),
                    opacity=0.9,
                ),
                customdata=np.c_[
                    df["distance"].to_numpy(dtype=float),
                    df[speed_col].to_numpy(dtype=float)
                ],
                hovertemplate=(
                    "dist: %{customdata[0]:.1f}m<br>"
                    "speed: %{customdata[1]:.2f} km/h"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"{track_name} — {speed_col} Track Map (lap_id={lap_id})",
            template="plotly_white",
            xaxis_title="x",
            yaxis_title="y",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        out_path = out_dir / f"{track_name}_predicted_speed_map.html"
        pio.write_html(fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn")
        return out_path

    @staticmethod
    def plot_car_state(
        laps: pd.DataFrame,
        track_name: str,
        out_dir: Path,
        brake_threshold: float = 0.4,
        throttle_threshold: float = 0.4,
        bin_m: float = 5.0,
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        track_df = laps.loc[laps["track"].astype(str) == str(track_name)].copy()
        if track_df.empty:
            return out_dir / f"{track_name}_car_state.html"

        brake_col = "brake" if "brake" in track_df.columns else "y_brake_zone"
        throttle_col = "throttle" if "throttle" in track_df.columns else "y_throttle_zone"
        
        p_brake_col = "p_brake_zone" if "p_brake_zone" in track_df.columns else brake_col
        p_throttle_col = "p_throttle_zone" if "p_throttle_zone" in track_df.columns else throttle_col

        track_df["cl_bin"] = (track_df["cl_dist"] / bin_m).round().astype(int) * bin_m
        agg = (
            track_df.groupby("cl_bin", as_index=False)
            .agg(x=("x", "mean"), y=("y", "mean"),
                brake=(brake_col, "mean"),
                throttle=(throttle_col, "mean"),
                p_brake=(p_brake_col, "mean"),
                p_throttle=(p_throttle_col, "mean"),
                cl_dist=("cl_dist", "mean"))
            .sort_values("cl_bin")
            .reset_index(drop=True)
        )

        brake_val    = agg["brake"].to_numpy(dtype=float)
        throttle_val = agg["throttle"].to_numpy(dtype=float)
        p_brake      = agg["p_brake"].to_numpy(dtype=float)
        p_throttle   = agg["p_throttle"].to_numpy(dtype=float)
        x  = agg["x"].to_numpy()
        y  = agg["y"].to_numpy()
        cl = agg["cl_dist"].to_numpy()

        # Use the physical inputs to decide the map nodes
        brake_mask    = brake_val >= brake_threshold
        throttle_mask = (~brake_mask) & (throttle_val >= throttle_threshold)
        corner_mask   = (~brake_mask) & (~throttle_mask)

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="lines",
            line=dict(color="rgba(0,0,0,0.15)", width=10),
            name="Centreline", hoverinfo="skip",
        ))
        if corner_mask.any():
            fig.add_trace(go.Scattergl(
                x=x[corner_mask], y=y[corner_mask], mode="markers",
                name="Cornering",
                marker=dict(size=5, color="rgba(160,160,160,0.7)"),
                customdata=np.c_[cl[corner_mask], brake_val[corner_mask], throttle_val[corner_mask], p_brake[corner_mask], p_throttle[corner_mask]],
                hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>brake=%{customdata[1]:.3f} (p=%{customdata[3]:.3f})<br>throttle=%{customdata[2]:.3f} (p=%{customdata[4]:.3f})<extra></extra>",
            ))
        if throttle_mask.any():
            fig.add_trace(go.Scattergl(
                x=x[throttle_mask], y=y[throttle_mask], mode="markers",
                name="Throttle zone",
                marker=dict(size=6, color="rgba(34,180,34,0.9)"),
                customdata=np.c_[cl[throttle_mask], throttle_val[throttle_mask], p_throttle[throttle_mask]],
                hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>throttle=%{customdata[1]:.3f} (p=%{customdata[2]:.3f})<extra></extra>",
            ))
        if brake_mask.any():
            fig.add_trace(go.Scattergl(
                x=x[brake_mask], y=y[brake_mask], mode="markers",
                name="Brake zone",
                marker=dict(size=7, color="rgba(220,20,20,0.95)"),
                customdata=np.c_[cl[brake_mask], brake_val[brake_mask], p_brake[brake_mask]],
                hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>brake=%{customdata[1]:.3f} (p=%{customdata[2]:.3f})<extra></extra>",
            ))

        fig.update_layout(
            title=f"{track_name} — braking / throttle zones",
            template="plotly_white",
            xaxis_title="x", yaxis_title="y",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        out_path = out_dir / f"{track_name}_car_state.html"
        pio.write_html(fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn")
        return out_path

    @staticmethod
    def plot_curvature_and_speed_dual_axis(
        laps: pd.DataFrame,
        track_name: str,
        out_dir: Path,
        curv_col: str = "c_smooth",
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = laps.loc[laps["track"].astype(str) == str(track_name)].copy()
        if df.empty:
            return out_dir / f"{track_name}_dual_axis.html"
            
        # Get just one lap to keep the graph readable
        lap_id = str(df["lap_id"].astype(str).iloc[0])
        df = df.loc[df["lap_id"].astype(str) == lap_id].sort_values("distance")

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        distance = df["distance"].to_numpy()
        speed = df["speed"].to_numpy()
        curvature = df[curv_col].to_numpy()

        # Trace 0: Speed (Primary Y)
        fig.add_trace(
            go.Scatter(
                x=distance, y=speed,
                mode="lines", name="Speed (km/h)",
                line=dict(color="blue", width=2)
            ),
            secondary_y=False,
        )

        # Trace 1: Curvature (Secondary Y)
        fig.add_trace(
            go.Scatter(
                x=distance, y=curvature,
                mode="lines", name=f"Curvature ({curv_col})",
                line=dict(color="red", width=2)
            ),
            secondary_y=True,
        )

        # Build Dropdown menus
        updatemenus = [
            dict(
                active=0,
                buttons=list([
                    dict(
                        label="Both",
                        method="update",
                        args=[{"visible": [True, True]},
                              {"title": f"{track_name} — Speed and Curvature"}]
                    ),
                    dict(
                        label="Speed Only",
                        method="update",
                        args=[{"visible": [True, False]},
                              {"title": f"{track_name} — Speed Only"}]
                    ),
                    dict(
                        label="Curvature Only",
                        method="update",
                        args=[{"visible": [False, True]},
                              {"title": f"{track_name} — Curvature Only"}]
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1, xanchor="left",
                y=1.15, yanchor="top"
            )
        ]

        fig.update_layout(
            title=f"{track_name} — Speed and Curvature (lap_id={lap_id})",
            template="plotly_white",
            updatemenus=updatemenus,
            xaxis_title="Distance (m)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.update_yaxes(title_text="Speed (km/h)", secondary_y=False, color="blue")
        fig.update_yaxes(title_text=f"Curvature ({curv_col})", secondary_y=True, color="red")

        out_path = out_dir / f"{track_name}_dual_axis.html"
        pio.write_html(fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn")
        return out_path
    
    @staticmethod
    def plot_car_state_3d(
        laps: pd.DataFrame,
        track_name: str,
        out_dir: Path,
        brake_threshold: float = 0.4,
        throttle_threshold: float = 0.4,
        bin_m: float = 5.0,
        z_col: str = "z"
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        track_df = laps.loc[laps["track"].astype(str) == str(track_name)].copy()
        if track_df.empty:
            return out_dir / f"{track_name}_car_state_3d.html"

        brake_col = "brake" if "brake" in track_df.columns else "y_brake_zone"
        throttle_col = "throttle" if "throttle" in track_df.columns else "y_throttle_zone"
        
        p_brake_col = "p_brake_zone" if "p_brake_zone" in track_df.columns else brake_col
        p_throttle_col = "p_throttle_zone" if "p_throttle_zone" in track_df.columns else throttle_col

        track_df["cl_bin"] = (track_df["cl_dist"] / bin_m).round().astype(int) * bin_m
        
        agg = (
            track_df.groupby("cl_bin", as_index=False)
            .agg(x=("x", "mean"), y=("y", "mean"), z_val=(z_col, "mean"),
                brake=(brake_col, "mean"),
                throttle=(throttle_col, "mean"),
                p_brake=(p_brake_col, "mean"),
                p_throttle=(p_throttle_col, "mean"),
                cl_dist=("cl_dist", "mean"))
            .sort_values("cl_bin")
            .reset_index(drop=True)
        )

        brake_val    = agg["brake"].to_numpy(dtype=float)
        throttle_val = agg["throttle"].to_numpy(dtype=float)
        p_brake      = agg["p_brake"].to_numpy(dtype=float)
        p_throttle   = agg["p_throttle"].to_numpy(dtype=float)
        x  = agg["x"].to_numpy()
        y  = agg["y"].to_numpy()
        z  = agg["z_val"].to_numpy()
        cl = agg["cl_dist"].to_numpy()

        brake_mask    = brake_val >= brake_threshold
        throttle_mask = (~brake_mask) & (throttle_val >= throttle_threshold)
        corner_mask   = (~brake_mask) & (~throttle_mask)

        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color="rgba(0,0,0,0.15)", width=4),
            name="Centreline", hoverinfo="skip",
        ))
        
        if corner_mask.any():
            fig.add_trace(go.Scatter3d(
                x=x[corner_mask], y=y[corner_mask], z=z[corner_mask], mode="markers",
                name="Cornering",
                marker=dict(size=4, color="rgba(160,160,160,0.7)"),
                customdata=np.c_[cl[corner_mask], brake_val[corner_mask], throttle_val[corner_mask], p_brake[corner_mask], p_throttle[corner_mask]],
                hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>brake=%{customdata[1]:.3f} (p=%{customdata[3]:.3f})<br>throttle=%{customdata[2]:.3f} (p=%{customdata[4]:.3f})<extra></extra>",
            ))
            
        if throttle_mask.any():
            fig.add_trace(go.Scatter3d(
                x=x[throttle_mask], y=y[throttle_mask], z=z[throttle_mask], mode="markers",
                name="Throttle zone",
                marker=dict(size=5, color="rgba(34,180,34,0.9)"),
                customdata=np.c_[cl[throttle_mask], throttle_val[throttle_mask], p_throttle[throttle_mask]],
                hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>throttle=%{customdata[1]:.3f} (p=%{customdata[2]:.3f})<extra></extra>",
            ))
            
        if brake_mask.any():
            fig.add_trace(go.Scatter3d(
                x=x[brake_mask], y=y[brake_mask], z=z[brake_mask], mode="markers",
                name="Brake zone",
                marker=dict(size=6, color="rgba(220,20,20,0.95)"),
                customdata=np.c_[cl[brake_mask], brake_val[brake_mask], p_brake[brake_mask]],
                hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>brake=%{customdata[1]:.3f} (p=%{customdata[2]:.3f})<extra></extra>",
            ))

        fig.update_layout(
            title=f"{track_name} — 3D Braking / Throttle Zones",
            template="plotly_white",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title=z_col.capitalize(),
                aspectmode='data'
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        out_path = out_dir / f"{track_name}_car_state_3d.html"
        pio.write_html(fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn")
        return out_path

    @staticmethod
    def plot_track_dashboard(
        laps: pd.DataFrame,
        track_name: str,
        out_dir: Path,
        speed_col: str = "predicted_speed",
        curv_col: str = "c_smooth",
        brake_threshold: float = 0.4,
        throttle_threshold: float = 0.4,
        bin_m: float = 5.0,
        z_col: str = "z",
        z_exaggeration: float = 10.0,
    ) -> Path:
        from plotly.subplots import make_subplots

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = laps.loc[laps["track"].astype(str) == str(track_name)].copy()
        if df.empty:
            return out_dir / f"{track_name}_dashboard.html"

        df_sorted = df.sort_values(["lap_id", "distance"])
        lap_id = str(df_sorted["lap_id"].astype(str).iloc[0])
        base_lap = df_sorted.loc[df_sorted["lap_id"].astype(str) == lap_id]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        actual_speed_col = speed_col if speed_col in base_lap.columns else "speed"
        actual_z_col = z_col if z_col in base_lap.columns else actual_speed_col # fallback for z
        
        fig.add_trace(go.Scattergl(x=base_lap["x"], y=base_lap["y"], mode="lines", name="Track", line=dict(color="rgba(0,0,0,0.55)", width=2), hoverinfo="skip"), secondary_y=False)
        fig.add_trace(go.Scattergl(x=base_lap["x"], y=base_lap["y"], mode="markers", name=f"Speed ({actual_speed_col})", marker=dict(size=4, color=base_lap[actual_speed_col], colorscale="Turbo", showscale=True, colorbar=dict(title="km/h")), customdata=np.c_[base_lap["distance"], base_lap[actual_speed_col]], hovertemplate="dist: %{customdata[0]:.1f}m<br>speed: %{customdata[1]:.2f} km/h<extra></extra>"), secondary_y=False)

        brake_col = "brake" if "brake" in df.columns else "y_brake_zone"
        throttle_col = "throttle" if "throttle" in df.columns else "y_throttle_zone"
        p_brake_col = "p_brake_zone" if "p_brake_zone" in df.columns else brake_col
        p_throttle_col = "p_throttle_zone" if "p_throttle_zone" in df.columns else throttle_col

        base_lap_copy = base_lap.copy()
        base_lap_copy["cl_bin"] = (base_lap_copy["cl_dist"] / bin_m).round().astype(int) * bin_m
        
        agg = base_lap_copy.groupby("cl_bin", as_index=False).agg(
            x=("x", "mean"), y=("y", "mean"), z_val=(actual_z_col, "mean"),
            brake=(brake_col, "mean"), throttle=(throttle_col, "mean"), p_brake=(p_brake_col, "mean"), p_throttle=(p_throttle_col, "mean"), cl_dist=("cl_dist", "mean")
        ).sort_values("cl_bin").reset_index(drop=True)

        b_mask = agg["brake"] >= brake_threshold
        t_mask = (~b_mask) & (agg["throttle"] >= throttle_threshold)
        c_mask = (~b_mask) & (~t_mask)

        fig.add_trace(go.Scattergl(x=agg["x"], y=agg["y"], mode="lines", line=dict(color="rgba(0,0,0,0.15)", width=10), name="Centreline", hoverinfo="skip", visible=False), secondary_y=False)
        
        c_data = agg.loc[c_mask]
        fig.add_trace(go.Scattergl(x=c_data["x"], y=c_data["y"], mode="markers", name="Cornering", marker=dict(size=5, color="rgba(160,160,160,0.7)"), customdata=np.c_[c_data["cl_dist"], c_data["brake"], c_data["throttle"], c_data["p_brake"], c_data["p_throttle"]], hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>brake=%{customdata[1]:.3f} (p=%{customdata[3]:.3f})<br>throttle=%{customdata[2]:.3f} (p=%{customdata[4]:.3f})<extra></extra>", visible=False), secondary_y=False)
        
        t_data = agg.loc[t_mask]
        fig.add_trace(go.Scattergl(x=t_data["x"], y=t_data["y"], mode="markers", name="Throttle", marker=dict(size=6, color="rgba(34,180,34,0.9)"), customdata=np.c_[t_data["cl_dist"], t_data["throttle"], t_data["p_throttle"]], hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>throttle=%{customdata[1]:.3f} (p=%{customdata[2]:.3f})<extra></extra>", visible=False), secondary_y=False)
        
        b_data = agg.loc[b_mask]
        fig.add_trace(go.Scattergl(x=b_data["x"], y=b_data["y"], mode="markers", name="Brake", marker=dict(size=7, color="rgba(220,20,20,0.95)"), customdata=np.c_[b_data["cl_dist"], b_data["brake"], b_data["p_brake"]], hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>brake=%{customdata[1]:.3f} (p=%{customdata[2]:.3f})<extra></extra>", visible=False), secondary_y=False)

        fig.add_trace(go.Scatter(x=base_lap["distance"], y=base_lap[curv_col], mode="lines", name=f"Curvature ({curv_col})", line=dict(color="red", width=2), visible=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=base_lap["distance"], y=base_lap["speed"], mode="lines", name="Speed (km/h)", line=dict(color="blue", width=2), visible=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=base_lap["distance"], y=base_lap[curv_col], mode="lines", name=f"Curvature ({curv_col})", line=dict(color="red", width=2), visible=False), secondary_y=True)

        # Struts for 3D Car State:
        z_min = agg["z_val"].min()
        x_struts, y_struts, z_struts = [], [], []
        for x_p, y_p, z_p in zip(agg["x"][::2], agg["y"][::2], agg["z_val"][::2]):
            x_struts.extend([x_p, x_p, None])
            y_struts.extend([y_p, y_p, None])
            z_struts.extend([z_p, z_min, None])
            
        fig.add_trace(go.Scatter3d(x=x_struts, y=y_struts, z=z_struts, mode="lines", line=dict(color="rgba(0,0,0,0.1)", width=2), name="Supports", hoverinfo="skip", visible=False))
        fig.add_trace(go.Scatter3d(x=agg["x"], y=agg["y"], z=agg["z_val"], mode="lines", line=dict(color="rgba(0,0,0,0.3)", width=4), name="Centreline (3D)", hoverinfo="skip", visible=False))
        fig.add_trace(go.Scatter3d(x=c_data["x"], y=c_data["y"], z=c_data["z_val"], mode="markers", name="Cornering (3D)", marker=dict(size=3, color="rgba(160,160,160,0.7)"), customdata=np.c_[c_data["cl_dist"], c_data["brake"], c_data["throttle"], c_data["p_brake"], c_data["p_throttle"]], hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>brake=%{customdata[1]:.3f} (p=%{customdata[3]:.3f})<br>throttle=%{customdata[2]:.3f} (p=%{customdata[4]:.3f})<extra></extra>", visible=False))
        fig.add_trace(go.Scatter3d(x=t_data["x"], y=t_data["y"], z=t_data["z_val"], mode="markers", name="Throttle (3D)", marker=dict(size=4, color="rgba(34,180,34,0.9)"), customdata=np.c_[t_data["cl_dist"], t_data["throttle"], t_data["p_throttle"]], hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>throttle=%{customdata[1]:.3f} (p=%{customdata[2]:.3f})<extra></extra>", visible=False))
        fig.add_trace(go.Scatter3d(x=b_data["x"], y=b_data["y"], z=b_data["z_val"], mode="markers", name="Brake (3D)", marker=dict(size=5, color="rgba(220,20,20,0.95)"), customdata=np.c_[b_data["cl_dist"], b_data["brake"], b_data["p_brake"]], hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>brake=%{customdata[1]:.3f} (p=%{customdata[2]:.3f})<extra></extra>", visible=False))

        updatemenus = [dict(
            active=0,
            buttons=[
                dict(label="Predicted Speed Map", method="update", args=[
                    {"visible": [True, True, False, False, False, False, False, False, False, False, False, False, False, False]},
                    {
                        "title.text": f"{track_name} — Predicted Speed", "xaxis.title.text": "x", "yaxis.title.text": "y", 
                        "yaxis.scaleanchor": "x", "yaxis2.visible": False,
                        "xaxis.visible": True, "yaxis.visible": True,
                        "scene.domain.x": [0.0, 0.001], "scene.domain.y": [0.0, 0.001] 
                    }
                ]),
                dict(label="Car State Map", method="update", args=[
                    {"visible": [False, False, True, True, True, True, False, False, False, False, False, False, False, False]},
                    {
                        "title.text": f"{track_name} — Car State", "xaxis.title.text": "x", "yaxis.title.text": "y", 
                        "yaxis.scaleanchor": "x", "yaxis2.visible": False,
                        "xaxis.visible": True, "yaxis.visible": True,
                        "scene.domain.x": [0.0, 0.001], "scene.domain.y": [0.0, 0.001]
                    }
                ]),
                dict(label="Car State Map (3D)", method="update", args=[
                    {"visible": [False, False, False, False, False, False, False, False, False, True, True, True, True, True]},
                    {
                        "title.text": f"{track_name} — 3D Car State", "xaxis.title.text": "", "yaxis.title.text": "", 
                        "yaxis2.visible": False,
                        "xaxis.visible": False, "yaxis.visible": False,
                        "scene.domain.x": [0.0, 1.0], "scene.domain.y": [0.0, 1.0]
                    }
                ]),
                dict(label="Curvature over Distance", method="update", args=[
                    {"visible": [False, False, False, False, False, False, True, False, False, False, False, False, False, False]},
                    {
                        "title.text": f"{track_name} — Curvature over Distance", "xaxis.title.text": "Distance (m)", "yaxis.title.text": f"Curvature ({curv_col})", 
                        "yaxis.scaleanchor": None, "yaxis2.visible": False,
                        "xaxis.visible": True, "yaxis.visible": True,
                        "scene.domain.x": [0.0, 0.001], "scene.domain.y": [0.0, 0.001] 
                    }
                ]),
                dict(label="Dual Axis (Dist)", method="update", args=[
                    {"visible": [False, False, False, False, False, False, False, True, True, False, False, False, False, False]},
                    {
                        "title.text": f"{track_name} — Distance Dual Axis", "xaxis.title.text": "Distance (m)", "yaxis.title.text": "Speed (km/h)", 
                        "yaxis.scaleanchor": None, "yaxis2.visible": True,
                        "xaxis.visible": True, "yaxis.visible": True,
                        "scene.domain.x": [0.0, 0.001], "scene.domain.y": [0.0, 0.001] 
                    }
                ]),
            ],
            direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.0, xanchor="left", y=1.2, yanchor="top"
        )]

        x_span = agg["x"].max() - agg["x"].min()
        y_span = agg["y"].max() - agg["y"].min()
        z_span = agg["z_val"].max() - agg["z_val"].min()
        if z_span == 0: 
            z_span = 1.0

        max_span = max(x_span, y_span)
        true_z_ratio = z_span / max_span
        dynamic_z_aspect = min(true_z_ratio * z_exaggeration, 0.4)

        fig.update_layout(
            title=f"{track_name} — Predicted Speed",
            template="plotly_white",
            updatemenus=updatemenus,
            xaxis_title="x",
            yaxis_title="y",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            scene=dict(
                domain=dict(x=[0.0, 0.001], y=[0.0, 0.001]),
                aspectmode='manual',
                aspectratio=dict(
                    x=x_span / max_span, 
                    y=y_span / max_span, 
                    z=dynamic_z_aspect 
                ),
                camera=dict(
                    eye=dict(x=0.0, y=-1.5, z=1.2) 
                ),
                zaxis_title=actual_z_col.capitalize()
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
            margin=dict(t=120, b=0, l=0, r=0)
        )
        
        fig.update_yaxes(title_text=f"Curvature ({curv_col})", secondary_y=True)
        fig.layout.yaxis2.visible = False

        out_path = out_dir / f"{track_name}_dashboard.html"
        pio.write_html(fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn")
        return out_path