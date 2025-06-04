from moviepy.editor import VideoFileClip
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_extract_features(video_path, audio_path):
    try:
        logger.debug(f"Extraindo áudio de {video_path} para {audio_path}")
        video = VideoFileClip(video_path)
        if video.duration > 600:
            logger.warning(f"Vídeo {video_path} muito longo ({video.duration}s). Limitando a 10 minutos.")
            video = video.subclip(0, 600)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=['-ar', '44100'])
        video.close()
        if not os.path.exists(audio_path):
            logger.error(f"Falha ao criar arquivo de áudio: {audio_path}")
            raise FileNotFoundError(f"Arquivo de áudio não criado: {audio_path}")
        logger.debug(f"Áudio extraído com sucesso: {audio_path}")
        return {"audio_path": audio_path}
    except Exception as e:
        logger.error(f"Erro ao extrair áudio de {video_path}: {str(e)}")
        raise