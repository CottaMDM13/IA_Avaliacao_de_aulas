import whisper
import os
import json
from datetime import datetime

def transcribe_audio(audio_path, output_dir):
    try:
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
        return {"text": result["text"], "txt_path": txt_path, "json_path": json_path}
    except Exception as e:
        raise ValueError(f"Erro ao transcrever áudio: {e}")
