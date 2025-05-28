import yaml
from src.utils.video_loader import load_and_extract_features
from src.utils.transcriber import transcribe_audio
from src.analysis.video import analyze_video, evaluate_video_quality
from src.analysis.audio import extract_audio_features, evaluate_audio_quality
from src.analysis.text import analyze_transcription
from src.analysis.metrics import calculate_overall_score
from src.utils.report import generate_final_report

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    video_path = f"{config['paths']['input_videos']}/aula_exemplo.mp4"
    audio_path = f"{config['paths']['input_audio']}/audio.wav"
    output_dir = config["paths"]["output_reports"]

    try:
        # Extrair áudio e features de vídeo
        video_features = load_and_extract_features(video_path, audio_path)
        print("Vídeo processado:", video_features)

        # Transcrever áudio
        transcription = transcribe_audio(audio_path, config["paths"]["output_transcripts"])
        print("Transcrição salva:", transcription["txt_path"])

        # Analisar vídeo
        video_metrics = analyze_video(video_path)
        video_results = evaluate_video_quality(video_metrics, config)
        print("Análise de vídeo concluída:", video_results["overall_score"])

        # Analisar áudio
        audio_features = extract_audio_features(audio_path)
        audio_results = evaluate_audio_quality(audio_features, config)
        print("Análise de áudio concluída:", audio_results["quality_score"])

        # Analisar texto
        text_results = analyze_transcription(transcription["text"], config)
        print("Análise de texto concluída:", text_results.get("overall_score", 0))

        # Calcular score geral
        metrics = calculate_overall_score(video_results, audio_results, text_results, config)
        print("Score geral:", metrics["overall_score"], "Aprovado:", metrics["approved"])

        # Gerar relatório final
        report = generate_final_report(video_results, audio_results, text_results, metrics, output_dir)
        print("Relatório final gerado:", report["json_path"])

    except Exception as e:
        print(f"Erro no pipeline: {e}")

if __name__ == "__main__":
    main()