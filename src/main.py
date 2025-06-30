import yaml
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.video_loader import load_and_extract_features
from src.utils.transcriber import transcribe_audio
from src.analysis.video import analyze_video, evaluate_video_quality
from src.analysis.audio import extract_audio_features, evaluate_audio_quality
from src.analysis.text import analyze_transcriptions
from src.analysis.metrics import calculate_overall_score
from src.utils.report import generate_final_report

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    try:
        with open("config/settings.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar config/settings.yaml: {str(e)}")
        raise

def process_video(video_file, config):
    start_time = time.time()
    video_path = f"{config['paths']['input_videos']}/{video_file}"
    audio_path = f"{config['paths']['input_audio']}/{video_file.rsplit('.', 1)[0]}.wav"  # Linha 29: Removido espaço extra
    output_dir = config["paths"]["output_reports"]

    try:
        logger.info(f"Iniciando processamento de: {video_file}")
        
        # Processar vídeo, áudio e transcrição em paralelo
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_video_features = executor.submit(load_and_extract_features, video_path, audio_path)
            future_transcription = executor.submit(transcribe_audio, audio_path, config["paths"]["output_transcripts"])
            future_video_metrics = executor.submit(analyze_video, video_path)
            
            video_features = future_video_features.result()
            logger.info(f"Vídeo processado: {video_file}, {video_features}")
            
            transcription = future_transcription.result()
            logger.info(f"Transcrição salva: {transcription['txt_path']}")
            
            video_metrics = future_video_metrics.result()
            logger.info(f"Análise de vídeo concluída")

        # Avaliar qualidade do vídeo
        video_results = evaluate_video_quality(video_metrics, config)
        logger.info(f"Avaliação de vídeo: {video_results['overall_score']}")

        # Processar áudio
        audio_features = extract_audio_features(audio_path)
        audio_results = evaluate_audio_quality(audio_features, config)
        logger.info(f"Análise de áudio concluída: {audio_results['quality_score']}")

        # Analisar texto (didática e geral)
        text_results = analyze_transcriptions([transcription["text"]], [video_results], [audio_results], config)[0]
        logger.info(f"Análise de texto concluída: Didática={text_results['didactic_score']}, Geral={text_results['overall_score']}")

        # Calcular score geral
        metrics = calculate_overall_score(video_results, audio_results, text_results, config)
        logger.info(f"Score geral: {metrics['overall_score']} Aprovado: {metrics['approved']}")

        # Gerar relatório final
        report = generate_final_report(video_results, audio_results, text_results, metrics, output_dir, video_file)
        logger.info(f"Relatório final gerado: {report['json_path']}")
        logger.info(f"Tempo de processamento de {video_file}: {time.time() - start_time:.2f} segundos")

    except Exception as e:
        logger.error(f"Erro ao processar {video_file}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    config = load_config()
    video_dir = config["paths"]["input_videos"]
    
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    
    if not video_files:
        logger.warning(f"Nenhum vídeo encontrado em {video_dir}")
        return
    
    logger.info(f"Processando {len(video_files)} vídeo(s)...")
    for video_file in video_files:
        process_video(video_file, config)

if __name__ == "__main__":
    main()