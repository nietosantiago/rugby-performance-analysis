"""
Rugby Analytics Dashboard — Streamlit

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ───────────────────────────────────────────────
# Page configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Rugby Analytics Dashboard",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────
# Data loading
# ───────────────────────────────────────────────

@st.cache_data
def load_events() -> pd.DataFrame:
    for path in [
        os.path.join(config.PROCESSED_DIR, "events.csv"),
        os.path.join(config.OUTPUT_DIR, "events.csv"),
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_player_stats() -> pd.DataFrame:
    for path in [
        os.path.join(config.PROCESSED_DIR, "player_stats.csv"),
        os.path.join(config.OUTPUT_DIR, "players.csv"),
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_team_stats() -> pd.DataFrame:
    for path in [
        os.path.join(config.PROCESSED_DIR, "team_stats.csv"),
        os.path.join(config.OUTPUT_DIR, "teams.csv"),
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


events_df = load_events()
player_stats = load_player_stats()
team_stats = load_team_stats()

has_data = not events_df.empty

# ───────────────────────────────────────────────
# Sidebar — Filters
# ───────────────────────────────────────────────

st.sidebar.title("🏉 Filtros")

if has_data:
    all_teams = sorted(events_df["team"].dropna().unique().tolist())
    selected_team = st.sidebar.selectbox("Equipo", ["Todos"] + all_teams)

    if selected_team != "Todos":
        player_pool = events_df[events_df["team"] == selected_team]["player_id"].unique()
    else:
        player_pool = events_df["player_id"].unique()
    selected_player = st.sidebar.selectbox(
        "Jugador", ["Todos"] + sorted(player_pool.tolist())
    )

    all_event_types = sorted(events_df["event_type"].dropna().unique().tolist())
    selected_events = st.sidebar.multiselect(
        "Tipo de evento", all_event_types, default=all_event_types
    )
else:
    selected_team = "Todos"
    selected_player = "Todos"
    selected_events = []

# Apply filters
filtered = events_df.copy()
if selected_team != "Todos":
    filtered = filtered[filtered["team"] == selected_team]
if selected_player != "Todos":
    filtered = filtered[filtered["player_id"] == selected_player]
if selected_events:
    filtered = filtered[filtered["event_type"].isin(selected_events)]

# ───────────────────────────────────────────────
# Header
# ───────────────────────────────────────────────

st.title("🏉 Dashboard de Análisis de Rugby")
st.markdown("Analíticas avanzadas y tracking de rendimiento de jugadores")

if not has_data:
    st.warning(
        "No se encontraron datos. Ejecuta `python main.py` para procesar un video."
    )
    st.stop()

# ───────────────────────────────────────────────
# KPIs
# ───────────────────────────────────────────────

st.header("📊 KPIs Principales")
k1, k2, k3, k4 = st.columns(4)

total_tackles = int((filtered["event_type"] == "Tackle").sum())
total_carries = int((filtered["event_type"] == "Carry").sum())
total_rucks = int((filtered["event_type"] == "Ruck").sum())
total_kicks = int((filtered["event_type"] == "Kick").sum())

k1.metric("Tackles", total_tackles)
k2.metric("Carries", total_carries)
k3.metric("Rucks", total_rucks)
k4.metric("Kicks", total_kicks)

# Second row
k5, k6, k7, k8 = st.columns(4)
n_players = filtered["player_id"].nunique()
n_events = len(filtered)

# Possession (if team stats available)
if not team_stats.empty and "total_possession_time" in team_stats.columns:
    poss_a = team_stats.loc[
        team_stats["team"] == config.TEAM_A_LABEL, "total_possession_time"
    ]
    poss_str = f"{float(poss_a.iloc[0]):.0f} %" if not poss_a.empty else "N/A"
else:
    poss_str = "N/A"

k5.metric("Jugadores Activos", n_players)
k6.metric("Total Eventos", n_events)
k7.metric(f"Posesión {config.TEAM_A_LABEL}", poss_str)
k8.metric("Lineouts", int((filtered["event_type"] == "Lineout").sum()))

# ───────────────────────────────────────────────
# Bar charts
# ───────────────────────────────────────────────

st.header("📈 Estadísticas por Evento")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Eventos por Jugador")
    if not filtered.empty:
        by_player = (
            filtered.groupby(["player_id", "event_type"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            by_player,
            x="player_id",
            y="count",
            color="event_type",
            barmode="group",
            title="Eventos por Jugador",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sin datos para graficar.")

with col_b:
    st.subheader("Eventos por Equipo")
    if not filtered.empty:
        by_team = (
            filtered.groupby(["team", "event_type"])
            .size()
            .reset_index(name="count")
        )
        fig2 = px.bar(
            by_team,
            x="team",
            y="count",
            color="event_type",
            barmode="stack",
            title="Distribución por Equipo",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Sin datos para graficar.")

# ───────────────────────────────────────────────
# Comparative player table
# ───────────────────────────────────────────────

st.header("👥 Tabla Comparativa de Jugadores")
if not player_stats.empty:
    display_ps = player_stats.copy()
    if selected_team != "Todos":
        display_ps = display_ps[display_ps["team"] == selected_team]
    if selected_player != "Todos":
        display_ps = display_ps[display_ps["player_id"] == selected_player]

    st.dataframe(
        display_ps.style.format({"meters_gained": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No hay estadísticas de jugadores disponibles.")

# ───────────────────────────────────────────────
# Helper: draw rugby pitch lines on a Plotly figure
# ───────────────────────────────────────────────

def add_rugby_field(fig: go.Figure) -> go.Figure:
    """Add pitch outline and key lines to a Plotly figure."""
    line_kw = dict(line_color="white", line_width=1, opacity=0.5)
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, **line_kw)
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line_dash="dash", **line_kw)
    fig.add_shape(type="line", x0=22, y0=0, x1=22, y1=100, line_dash="dot", **line_kw)
    fig.add_shape(type="line", x0=78, y0=0, x1=78, y1=100, line_dash="dot", **line_kw)
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Largo de Cancha (m)"),
        yaxis=dict(range=[0, 100], title="Ancho de Cancha (m)"),
        template="plotly_dark",
        plot_bgcolor="#2d5a27",
    )
    return fig


# ───────────────────────────────────────────────
# Heatmaps
# ───────────────────────────────────────────────

st.header("🔥 Mapas de Calor")
st.markdown(
    """
Distribución espacial de eventos en el campo de juego.
- **X**: Largo de la cancha (0 a 100 m)
- **Y**: Ancho de la cancha (0 a 100 m)
"""
)

hm1, hm2 = st.columns(2)

with hm1:
    st.subheader("Posición de Eventos")
    if not filtered.empty and "x" in filtered.columns and "y" in filtered.columns:
        # Try to show pre-generated PNG first
        png_shown = False
        if selected_team != "Todos" and len(selected_events) == 1:
            png_path = os.path.join(
                config.FIGURES_DIR,
                f"heatmap_{selected_events[0]}_{selected_team}.png",
            )
            if os.path.exists(png_path):
                st.image(png_path, use_container_width=True)
                png_shown = True

        if not png_shown:
            fig_hm = px.density_contour(
                filtered,
                x="x",
                y="y",
                title="Densidad de Eventos",
                color_discrete_sequence=["#ff6361"],
            )
            fig_hm.update_traces(contours_coloring="fill", colorscale="YlOrRd")
            fig_hm = add_rugby_field(fig_hm)
            st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No hay coordenadas disponibles.")

with hm2:
    st.subheader("Scatter de Eventos")
    if not filtered.empty and "x" in filtered.columns:
        fig_sc = px.scatter(
            filtered,
            x="x",
            y="y",
            color="event_type",
            symbol="team",
            hover_data=["player_id", "timestamp"],
            title="Ubicación de cada Evento",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_sc = add_rugby_field(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Sin datos de posición.")

# ───────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Dashboard generado automáticamente por el Sistema de Análisis de Video de Rugby"
)
