import yaml
import os
import time
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

def process_video(video_file, config):
    start_time = time.time()
    video_path = f"{config['paths']['input_videos']}/{video_file}"
    audio_path = f"{config['paths']['input_audio']}/{video_file.rsplit('.', 1)[0]}.wav"
    output_dir = config["paths"]["output_reports"]

    try:
        # Extrair áudio e features de vídeo
        video_features = load_and_extract_features(video_path, audio_path)
        print(f"Vídeo processado: {video_file}", video_features)

        # Transcrever áudio
        transcription = transcribe_audio(audio_path, config["paths"]["output_transcripts"])
        print(f"Transcrição salva: {transcription['txt_path']}")

        # Analisar vídeo
        video_metrics = analyze_video(video_path)
        video_results = evaluate_video_quality(video_metrics, config)
        print(f"Análise de vídeo concluída: {video_results['overall_score']}")

        # Analisar áudio
        audio_features = extract_audio_features(audio_path)
        audio_results = evaluate_audio_quality(audio_features, config)
        print(f"Análise de áudio concluída: {audio_results['quality_score']}")

        # Analisar texto com Gemini
        text_results = analyze_transcription(transcription["text"], video_results, audio_results, config)
        print(f"Análise de texto concluída: {text_results['overall_score']}")

        # Calcular score geral
        metrics = calculate_overall_score(video_results, audio_results, text_results, config)
        print(f"Score geral: {metrics['overall_score']} Aprovado: {metrics['approved']}")

        # Gerar relatório final
        report = generate_final_report(video_results, audio_results, text_results, metrics, output_dir, video_file)
        print(f"Relatório final gerado: {report['json_path']}")
        print(f"Tempo de processamento de {video_file}: {time.time() - start_time:.2f} segundos")

    except Exception as e:
        print(f"Erro ao processar {video_file}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    config = load_config()
    video_dir = config["paths"]["input_videos"]
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print("Nenhum vídeo encontrado em", video_dir)
        return
    
    print(f"Processando {len(video_files)} vídeo(s)...")
    for video_file in video_files:
        print(f"\nIniciando processamento de: {video_file}")
        process_video(video_file, config)

if __name__ == "__main__":
    main()