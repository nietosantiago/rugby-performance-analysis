import yaml
from video_processor import process_video
from metrics import calculate_metrics

def main():

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    video_folder = config["data"]["videos_folder"]

    print("Procesando videos...")

    process_video(video_folder)

    print("Calculando métricas...")

    calculate_metrics()

    print("Listo")

if __name__ == "__main__":
    main()