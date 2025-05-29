import whisper
import os
import json
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = None

def transcribe_audio(audio_path, output_dir):
    global model
    try:
        logger.info(f"Transcrevendo áudio: {audio_path}")
        if model is None:
            logger.debug("Carregando modelo Whisper")
            model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = f"{output_dir}/transcription_{timestamp}.txt"
        json_path = f"{output_dir}/transcription_{timestamp}.json"
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(result["text"])
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(result, f_json, indent=4, ensure_ascii=False)
        logger.info(f"Transcrição concluída: {txt_path}")
        return {"text": result["text"], "txt_path": txt_path, "json_path": json_path}
    except Exception as e:
        logger.error(f"Erro ao transcrever áudio {audio_path}: {str(e)}")
        raise ValueError(f"Erro ao transcrever áudio: {e}")
