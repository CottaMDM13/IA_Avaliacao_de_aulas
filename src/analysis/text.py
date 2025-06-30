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
        try:
            # Análise didática
            didactic_prompt = (
                "Analise a transcrição de uma aula com foco na didática, clareza, estrutura e engajamento. "
                "Gere um JSON com: 'didactic_review' (resumo narrativo, máx. 50 palavras), "
                "'didactic_score' (0-10), 'didactic_justification' (explicação curta). "
                "Use tom neutro e foco educacional.\n"
                f"Transcrição: {text}"
            )
            logger.debug("Enviando prompt para análise didática")
            didactic_response = model.generate_content(didactic_prompt)
            didactic_result = json.loads(didactic_response.text.strip("```json\n").strip("\n```"))

            # Análise geral
            general_prompt = (
                "Analise a transcrição, métricas de vídeo e áudio de uma aula. "
                "Gere um JSON com: 'review' (resumo narrativo, máx. 50 palavras), "
                "'overall_score' (0-10, baseado em didática, clareza, engajamento, vídeo e áudio), "
                "'justification' (explicação curta). Use tom neutro e foco educacional.\n"
                f"Transcrição: {text}\nMétricas de vídeo: {json.dumps(video_metrics, indent=2)}\n"
                f"Métricas de áudio: {json.dumps(audio_results, indent=2)}"
            )
            logger.debug("Enviando prompt para análise geral")
            general_response = model.generate_content(general_prompt)
            general_result = json.loads(general_response.text.strip("```json\n").strip("\n```"))

            result = {
                "didactic_review": didactic_result["didactic_review"],
                "didactic_score": didactic_result["didactic_score"],
                "didactic_justification": didactic_result["didactic_justification"],
                "review": general_result["review"],
                "overall_score": general_result["overall_score"],
                "justification": general_result["justification"]
            }
            logger.info(f"Análise de transcrição concluída: Didática={result['didactic_score']}, Geral={result['overall_score']}")
            results.append(result)
        except Exception as e:
            logger.error(f"Erro na análise de transcrição: {str(e)}")
            results.append({
                "didactic_review": "Erro na análise",
                "didactic_score": 0,
                "didactic_justification": f"Erro: {str(e)}",
                "review": "Erro na análise",
                "overall_score": 0,
                "justification": f"Erro: {str(e)}"
            })
    return results