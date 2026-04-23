import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

class PlotSessionProgression:
    @staticmethod
    def plot_laps(session_lap_summary: list, track_name: str, session_dir: Path):
        """
        Compare all session lap times and whether they're driven with/without advice overlay.
        """
        # Need at least 2 laps:
        if not session_lap_summary or len(session_lap_summary) < 2:
            return

        df_summary = pd.DataFrame(session_lap_summary)
        aided = df_summary[df_summary["overlay_active"] == True]
        unaided = df_summary[df_summary["overlay_active"] == False]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_summary["lap"],
            y=df_summary["time"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.3)", width=2, dash="dash"),
            name="Progression",
            hoverinfo="skip"
        ))

        if not unaided.empty:
            fig.add_trace(go.Scatter(
                x=unaided["lap"],
                y=unaided["time"],
                mode="markers",
                marker=dict(size=14, color="rgba(220, 20, 60, 0.9)", symbol="circle"),
                name="Baseline (Overlay OFF)",
                hovertemplate="Lap %{x}<br>Time: %{y:.3f}s<extra></extra>"
            ))

        if not aided.empty:
            fig.add_trace(go.Scatter(
                x=aided["lap"],
                y=aided["time"],
                mode="markers",
                marker=dict(size=14, color="rgba(34, 139, 34, 0.9)", symbol="diamond"),
                name="Aided (Overlay ON)",
                hovertemplate="Lap %{x}<br>Time: %{y:.3f}s<extra></extra>"
            ))

        # Red highlighted column on invalid laps:
        if "lap_invalid" in df_summary.columns:
            invalid_laps = df_summary[df_summary["lap_invalid"] == True]
            for _, row in invalid_laps.iterrows():
                lap_num = row["lap"]
                fig.add_vrect(
                    x0=lap_num - 0.05, 
                    x1=lap_num + 0.05, 
                    fillcolor="red", 
                    opacity=0.15, 
                    layer="above", 
                    line_width=0,
                    annotation_text="Track<br>Limits", 
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="rgba(255,0,0,0.6)"
                )

        fig.update_layout(
            title=f"Session Lap Time Progression ({track_name})",
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (Seconds)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        if session_dir:
            plot_path = session_dir / "session_lap_progression.html"
            fig.write_html(str(plot_path))
            print(f"Updated session lap time graph: {plot_path}")