import os
import google.generativeai as genai
import json
from dotenv import load_dotenv

def analyze_transcription(text, video_metrics, audio_results, config):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não definida.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = (
        "Você é um avaliador de aulas gravadas. Com base na transcrição, métricas de vídeo e áudio, gere um resumo narrativo (ex.: 'O professor demonstra domínio, mas a aula não é atrativa'). "
        "Avalie: domínio do assunto, didática, atratividade, coerência, continuidade, postura, gestos e tom de voz. "
        "Retorne um JSON com 'review' (resumo narrativo), 'overall_score' (0-10) e 'justification'.\n\n"
        f"Transcrição:\n{text}\n\n"
        f"Métricas de vídeo:\n{json.dumps(video_metrics, indent=2)}\n\n"
        f"Métricas de áudio:\n{json.dumps(audio_results, indent=2)}"
    )

    try:
        response = model.generate_content(prompt)
        return json.loads(response.text.strip("```json\n").strip("\n```"))
    except Exception as e:
        return {"review": "Erro na análise", "overall_score": 0, "justification": f"Erro: {str(e)}"}