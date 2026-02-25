import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import yaml

# Cargar configuracion
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Definir rutas
OUTPUT_DIR = "output"
PLAYERS_CSV = os.path.join(OUTPUT_DIR, "players.csv")
EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "metrics.csv")

# Función para cargar datos
@st.cache_data
def load_data():
    if not (os.path.exists(PLAYERS_CSV) and os.path.exists(EVENTS_CSV) and os.path.exists(METRICS_CSV)):
        # Si no existen, los generamos ejecutando el flujo
        from video_processor import process_video
        from metrics import MetricsCalculator
        from data_generator import DataGenerator
        from visualizer import Visualizer
        
        # Simula ejecucion si no hay csvs
        with st.spinner("Procesando videos y generando datos iniciales..."):
             video_folder = config["data"]["videos_folder"]
             events = process_video(video_folder)
             
             calculator = MetricsCalculator(events)
             players, teams = calculator.calculate()
             
             generator = DataGenerator(OUTPUT_DIR)
             generator.generate_events_csv(events)
             generator.generate_players_csv(players)
             generator.generate_teams_csv(teams)
             generator.generate_metrics_csv(players, teams)
             
             viz = Visualizer(OUTPUT_DIR)
             viz.generate_all_heatmaps(events)
             
    players_df = pd.read_csv(PLAYERS_CSV)
    events_df = pd.read_csv(EVENTS_CSV)
    metrics_df = pd.read_csv(METRICS_CSV)
    return players_df, events_df, metrics_df

st.set_page_config(page_title="Rugby Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

players_df = pd.DataFrame()
events_df = pd.DataFrame()
metrics_df = pd.DataFrame()

try:
    players_df, events_df, metrics_df = load_data()
except Exception as e:
    st.error(f"Error cargando los datos iniciales o procesando videos: {e}")
    st.stop()


# --- SIDEBAR FILTERS ---
st.sidebar.title("Filtros")

# Filtro por Partido (Asumiendo match_id en events_df)
available_matches = events_df['match_id'].unique() if not events_df.empty and 'match_id' in events_df.columns else ["N/A"]
selected_match = st.sidebar.selectbox("Seleccionar Partido", ["Todos"] + list(available_matches))


# Filtrar eventos según el partido
filtered_events = events_df if selected_match == "Todos" else events_df[events_df['match_id'] == selected_match]

# Filtro por Jugador (Basado en el equipo o en todos)
teams = players_df['team'].unique() if not players_df.empty else []
selected_team = st.sidebar.selectbox("Filtrar por Equipo", ["Todos"] + list(teams))

filtered_players = players_df if selected_team == "Todos" else players_df[players_df['team'] == selected_team]
player_list = filtered_players['player_id'].unique() if not filtered_players.empty else []
selected_player = st.sidebar.selectbox("Seleccionar Jugador", ["Todos"] + list(player_list))


# Aplicar filtrado final al dataframe de jugadores y metricas
if selected_player != "Todos":
    filtered_players = filtered_players[filtered_players['player_id'] == selected_player]
    filtered_metrics = metrics_df[(metrics_df['id'] == selected_player) & (metrics_df['entity_type'] == 'player')]
else:
    filtered_metrics = metrics_df[(metrics_df['team'] == selected_team) if selected_team != "Todos" else (metrics_df['entity_type'] == 'player')]

if selected_player != "Todos":
    filtered_events = filtered_events[filtered_events['player'] == selected_player]
elif selected_team != "Todos":
    filtered_events = filtered_events[filtered_events['team'] == selected_team]

st.title("🏉 Dashboard de Análisis de Rugby")
st.markdown("Analíticas avanzadas y tracking de rendimiento de jugadores")

# --- KPIs ---
st.header("Métricas Avanzadas")
cols = st.columns(4)

if not filtered_metrics.empty:
    # Usar promedios si hay multiples jugadores, o el valor exacto si es uno
    avg_tackle_eff = filtered_metrics['tackle_efficiency_pct'].mean()
    avg_meters_carry = filtered_metrics['meters_gained_per_carry'].mean()
    avg_rucks_won = filtered_metrics['rucks_won_pct'].mean()
    avg_participation = filtered_metrics['participation_index'].mean()

    cols[0].metric("Tackles Efectivos (%)", f"{avg_tackle_eff:.1f}%")
    cols[1].metric("Metros por Carry", f"{avg_meters_carry:.2f} m")
    cols[2].metric("Rucks Ganados (%)", f"{avg_rucks_won:.1f}%")
    cols[3].metric("Participación Promedio", f"{avg_participation:.1f}")
else:
    cols[0].metric("Tackles Efectivos (%)", "N/A")
    cols[1].metric("Metros por Carry", "N/A")
    cols[2].metric("Rucks Ganados (%)", "N/A")
    cols[3].metric("Participación Promedio", "N/A")


# --- GRAFICOS GENERALES ---
st.header("Estadísticas Generales")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Eventos por Jugador")
    # Agregar datos para Plotly
    if not filtered_players.empty:
        # Melt dataframe para graficar
        melted_df = pd.melt(filtered_players, id_vars=['player_id'], value_vars=['tackles', 'carries', 'rucks', 'lineouts', 'kicks'],
                            var_name='Event Type', value_name='Count')
        fig_events = px.bar(melted_df, x='player_id', y='Count', color='Event Type', barmode='group',
                            title="Desglose de Eventos")
        st.plotly_chart(fig_events, use_container_width=True)
    else:
        st.info("No hay datos de jugadores para los filtros seleccionados.")

with col2:
    st.subheader("Participación Global")
    if not filtered_players.empty:
        fig_pie = px.pie(filtered_players, values='participation_index', names='player_id', title="Índice de Participación por Jugador")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
       st.info("No hay datos para mostrar.")

# --- MAPAS DE CALOR Y POSICIONES ---
st.header("Análisis Espacial")

st.markdown("""
Los siguientes gráficos muestran la distribución espacial de los eventos en el campo de juego.
- **X**: Largo de la cancha (0 a 100m)
- **Y**: Ancho de la cancha (0 a 70m)
""")

# Función utilitaria para dibujar fondo de campo de rugby en plotly
def add_rugby_field(fig):
    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, title="Largo (m)"),
        yaxis=dict(range=[0, 70], showgrid=False, zeroline=False, title="Ancho (m)"),
        plot_bgcolor="#4CAF50",
    )
    # Lineas del campo
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=70, line=dict(color="white", width=2, dash="dash"))
    fig.add_shape(type="line", x0=22, y0=0, x1=22, y1=70, line=dict(color="white", width=2, dash="dot"))
    fig.add_shape(type="line", x0=78, y0=0, x1=78, y1=70, line=dict(color="white", width=2, dash="dot"))
    return fig

col_map1, col_map2 = st.columns(2)

with col_map1:
    st.subheader("Posición de Jugadores (Todos los eventos)")
    if not filtered_events.empty:
        fig_pos = px.density_heatmap(filtered_events, x="x", y="y", nbinsx=20, nbinsy=14, color_continuous_scale="Viridis", title="Mapa de Calor Global")
        fig_pos = add_rugby_field(fig_pos)
        st.plotly_chart(fig_pos, use_container_width=True)
    else:
        st.info("No hay eventos para mapear.")

with col_map2:
    st.subheader("Zonas de Tackle")
    tackle_events = filtered_events[filtered_events['event_type'] == 'Tackle']
    if not tackle_events.empty:
         fig_tck = px.scatter(tackle_events, x="x", y="y", color="team", size="minute", size_max=15, title="Distribución de Tackles")
         fig_tck = add_rugby_field(fig_tck)
         st.plotly_chart(fig_tck, use_container_width=True)
    else:
         st.info("No hay tackles registrados con estos filtros.")

st.subheader("Zonas de Ruck")
ruck_events = filtered_events[filtered_events['event_type'] == 'Ruck']
if not ruck_events.empty:
    fig_rck = px.density_contour(ruck_events, x="x", y="y", title="Densidad de Rucks")
    fig_rck.update_traces(contours_coloring="fill", colorscale="Reds")
    fig_rck = add_rugby_field(fig_rck)
    st.plotly_chart(fig_rck, use_container_width=True)
else:
    st.info("No hay rucks registrados con estos filtros.")

st.markdown("---")
st.caption("Dashboard creado automáticamente por el Sistema de Análisis de Video de Rugby")
