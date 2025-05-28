import whisper
import os
import json
from datetime import datetime

audio_path = "audio_processing/audio.wav"
txt_output_path = "transcripts_analysis/transcricao.txt"
json_output_path = "transcripts_analysis/transcricao.json"

try:
    model = whisper.load_model("base")  # pode mudar para "small", "medium", "large"
    result = model.transcribe(audio_path)

    os.makedirs("transcripts", exist_ok=True)

    # Salva o texto legível
    with open(txt_output_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(result["text"])

    # Salva o JSON completo
    with open(json_output_path, "w", encoding="utf-8") as f_json:
        json.dump(result, f_json, indent=4, ensure_ascii=False)

    print(f"Transcrição salva em:\n- {txt_output_path}\n- {json_output_path}")
except Exception as e:
    print(f"Erro ao transcrever áudio: {e}")
