from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

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
    def plot_braking_zones_by_track(
        laps: pd.DataFrame,
        out_dir: Path,
        prob_threshold: float = 0.5,
        zone_col: str = "p_brake_zone",
        max_zone_points: int = 15000,
    ) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[Path] = []
        for track_name, track_df in laps.groupby("track", sort=False):
            track_name = str(track_name)
            track_df = track_df.sort_values(["lap_id", "distance"])

            base_lap = PlotTrackMaps._pick_base_lap(track_df)
            lap_id = str(base_lap["lap_id"].astype(str).iloc[0])

            fig = go.Figure()
            PlotTrackMaps._add_track_line(fig, base_lap, lap_id=lap_id)

            zone_rows = PlotTrackMaps._select_zone_rows(
                track_df=track_df,
                zone_col=str(zone_col),
                prob_threshold=float(prob_threshold),
                max_points=int(max_zone_points),
            )
            PlotTrackMaps._add_zone_layer(fig, zone_rows, zone_col=str(zone_col), prob_threshold=float(prob_threshold))

            PlotTrackMaps._add_curvature_layer(fig, base_lap)

            fig.update_layout(
                title=f"{track_name} — {zone_col} overlay",
                template="plotly_white",
                xaxis_title="x",
                yaxis_title="y",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            out_path = out_dir / f"{track_name}_{zone_col}.html"
            pio.write_html(fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn")
            outputs.append(out_path)

        return outputs
    
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
                y=np.abs(df[curvature_col].to_numpy()),
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
    def plot_brake_density(
        gt: pd.DataFrame,
        track_name: str,
        out_dir: Path,
        prob_col: str = "p_brake_exp",
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig = go.Figure()

        # Centreline as background
        fig.add_trace(go.Scattergl(
            x=gt["x_exp"].to_numpy(), y=gt["y_exp"].to_numpy(),
            mode="lines",
            line=dict(color="rgba(0,0,0,0.2)", width=8),
            name="Centreline",
            hoverinfo="skip",
        ))

        # Density heatmap — color each centreline point by p_brake_exp
        fig.add_trace(go.Scattergl(
            x=gt["x_exp"].to_numpy(), y=gt["y_exp"].to_numpy(),
            mode="markers",
            name="Brake density",
            marker=dict(
                size=6,
                color=gt[prob_col].to_numpy(dtype=float),
                colorscale="RdYlGn_r",  # green=never, red=always
                cmin=0.0, cmax=1.0,
                showscale=True,
                colorbar=dict(title="P(brake)"),
            ),
            customdata=np.c_[gt["cl_dist"].to_numpy(), gt[prob_col].to_numpy()],
            hovertemplate="cl_dist=%{customdata[0]:.1f}m<br>p_brake=%{customdata[1]:.3f}<extra></extra>",
        ))

        fig.update_layout(
            title=f"{track_name} — brake density",
            template="plotly_white",
            xaxis_title="x", yaxis_title="y",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        out_path = out_dir / f"{track_name}_brake_density.html"
        pio.write_html(fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn")
        return out_path
