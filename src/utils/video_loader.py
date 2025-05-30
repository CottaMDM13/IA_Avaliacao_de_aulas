from moviepy.editor import VideoFileClip
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_extract_features(video_path, audio_output_path=None):
    try:
        logger.info(f"Extraindo features de vídeo: {video_path}")
        clip = VideoFileClip(video_path)
        features = {
            "duration_sec": float(clip.duration),
            "fps": clip.fps,
            "resolution": f"{clip.size[0]}x{clip.size[1]}"
        }
        if audio_output_path and clip.audio:
            os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
            clip.audio.write_audiofile(audio_output_path, codec='pcm_s16le')  # Linha 18: Adicionado codec
            features["audio_path"] = audio_output_path
            logger.debug(f"Áudio extraído: {audio_output_path}")
        clip.close()
        logger.info(f"Features extraídas: {video_path}")
        return features
    except Exception as e:
        logger.error(f"Erro ao processar vídeo {video_path}: {str(e)}")
        raise ValueError(f"Erro ao processar vídeo: {e}")