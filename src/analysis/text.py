import os
import google.generativeai as genai
import json
import logging
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache do modelo Gemini
_model = None

def get_gemini_model():
    global _model
    if _model is None:
        try:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY não definida")
                raise ValueError("GEMINI_API_KEY não definida.")
            genai.configure(api_key=api_key)
            _model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Modelo Gemini inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo Gemini: {str(e)}")
            raise
    return _model

def analyze_transcriptions(texts, video_metrics_list, audio_results_list, config):
    model = get_gemini_model()
    results = []
    for text, video_metrics, audio_results in zip(texts, video_metrics_list, audio_results_list):
        prompt = (
            "Analise a transcrição de uma aula e métricas fornecidas. Gere um JSON com: "
            "'review' (resumo narrativo, máx. 50 palavras), 'overall_score' (0-10, baseado em didática, clareza, engajamento), "
            "'justification' (explicação curta). Use tom neutro e foco educacional.\n"
            f"Transcrição: {text}\nMétricas de vídeo: {json.dumps(video_metrics, indent=2)}\n"
            f"Métricas de áudio: {json.dumps(audio_results, indent=2)}"
        )
        try:
            logger.debug("Enviando prompt para Gemini")
            response = model.generate_content(prompt)
            result = json.loads(response.text.strip("```json\n").strip("\n```"))
            logger.info(f"Análise de transcrição concluída: {result['overall_score']}")
            results.append(result)
        except Exception as e:
            logger.error(f"Erro na análise de transcrição: {str(e)}")
            results.append({"review": "Erro na análise", "overall_score": 0, "justification": f"Erro: {str(e)}"})
    return results
