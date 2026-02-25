import yaml
from video_processor import process_video
from metrics import MetricsCalculator

def main():

    # Leer configuración
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    video_folder = config["data"]["videos_folder"]

    print("Procesando videos...")

    # Procesar videos y obtener eventos
    events = process_video(video_folder)

    print("Calculando métricas...")

    # Crear calculador de métricas
    calculator = MetricsCalculator(events)

    players, teams = calculator.calculate()

    insights = calculator.get_insights()

    print("\n===== RESULTADOS =====")

    print("\nJugadores:")
    for p in players:
        print(p)

    print("\nEquipos:")
    for t in teams:
        print(t)

    print("\nInsights:")
    print(insights)

    print("\nListo")

if __name__ == "__main__":
    main()