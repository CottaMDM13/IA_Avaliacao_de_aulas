import librosa
import numpy as np
import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import json

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

def extract_audio_features(audio_path):
    try:
        logger.info(f"Extraindo features de áudio: {audio_path}")
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y)[0])
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        silence_ratio = np.mean(np.abs(y) < 0.01)
        pitch = np.mean(librosa.yin(y, fmin=65, fmax=2093))
        clipping = np.mean(np.abs(y) > 0.98)
        features = {
            "duration": duration,
            "rms": rms,
            "spectral_centroid": spectral_centroid,
            "silence_ratio": silence_ratio,
            "pitch": pitch,
            "clipping": clipping
        }
        logger.info(f"Features de áudio extraídas: {audio_path}")
        return features
    except Exception as e:
        logger.error(f"Erro ao extrair features de áudio {audio_path}: {str(e)}")
        raise

def evaluate_audio_quality(features, config):
    try:
        logger.debug("Avaliando qualidade do áudio")
        model = get_gemini_model()
        comments = []

        # Avaliar cada aspecto
        duration = features.get("duration", 0)
        duration_score = 1.0 if 300 <= duration <= 3600 else 0.5
        comments.append({"score": duration_score})

        clipping = features.get("clipping", 0)
        clipping_score = 0.2 if clipping > 0.1 else 0.95
        comments.append({"score": clipping_score})

        silence_ratio = features.get("silence_ratio", 0)
        silence_score = 0.95 if silence_ratio < 0.3 else 0.4
        comments.append({"score": silence_score})

        rms = features.get("rms", 0)
        volume_score = 0.95 if 0.01 < rms < 0.1 else 0.6
        comments.append({"score": volume_score})

        pitch = features.get("pitch", 0)
        pitch_score = 0.95 if 100 < pitch < 300 else 0.6
        comments.append({"score": pitch_score})

        # Gerar comentários via Gemini com prompt mais robusto
        prompt = (
            "Você é um especialista em análise de áudio educacional. Com base nas métricas de áudio fornecidas, gere comentários e sugestões para cada aspecto (duração, clipping, silêncio, volume, pitch) "
            "em um tom neutro, educacional e objetivo. Forneça um resumo narrativo (máx. 50 palavras) e uma sugestão de melhoria para cada aspecto. "
            "Retorne um JSON estruturado com 'comment' e 'suggestion' para cada aspecto, mesmo que os valores sejam genéricos. "
            "Formato esperado: ["
            "{\"comment\": \"string\", \"suggestion\": \"string\"}, "
            "{\"comment\": \"string\", \"suggestion\": \"string\"}, "
            "{\"comment\": \"string\", \"suggestion\": \"string\"}, "
            "{\"comment\": \"string\", \"suggestion\": \"string\"}, "
            "{\"comment\": \"string\", \"suggestion\": \"string\"}] "
            "Métricas fornecidas:\n"
            f"Duração: {duration:.2f} segundos (ideal: 300-3600s)\n"
            f"Clipping: {clipping:.4f} (ideal: <0.1)\n"
            f"Proporção de Silêncio: {silence_ratio:.4f} (ideal: <0.3)\n"
            f"Volume (RMS): {rms:.4f} (ideal: 0.01-0.1)\n"
            f"Pitch: {pitch:.2f} Hz (ideal: 100-300 Hz)\n"
            "Exemplo: ["
            "{\"comment\": \"Duração de 600s é adequada para aulas.\", \"suggestion\": \"Mantenha a duração dentro do ideal.\"}, "
            "{\"comment\": \"Clipping baixo, sem distorção.\", \"suggestion\": \"Continue monitorando o ganho.\"}]"
        )
        try:
            response = model.generate_content(prompt)
            logger.debug(f"Resposta bruta do Gemini: {response.text}")
            gemini_comments = json.loads(response.text.strip("```json\n").strip("\n```"))
            logger.debug(f"Resposta parseada do Gemini: {gemini_comments}")
            # Garantir que gemini_comments tenha 5 elementos
            if not isinstance(gemini_comments, list) or len(gemini_comments) != 5:
                logger.warning(f"Resposta do Gemini inválida, usando comentários de fallback")
                gemini_comments = [
                    {"comment": f"Duração {duration:.2f}s está {'adequada' if 300 <= duration <= 3600 else 'fora do ideal'}.", "suggestion": "Ajuste a duração para 5-60 minutos."},
                    {"comment": f"Clipping {clipping:.4f} {'excede' if clipping > 0.1 else 'está dentro do'} limite.", "suggestion": "Reduza o ganho para evitar clipping."},
                    {"comment": f"Silêncio {silence_ratio:.4f} {'é aceitável' if silence_ratio < 0.3 else 'é excessivo'}.", "suggestion": "Minimize pausas longas."},
                    {"comment": f"Volume RMS {rms:.4f} {'é adequado' if 0.01 < rms < 0.1 else 'está fora do ideal'}.", "suggestion": "Ajuste o volume para clareza."},
                    {"comment": f"Pitch {pitch:.2f} Hz {'é apropriado' if 100 < pitch < 300 else 'está fora do ideal'}.", "suggestion": "Module a voz para melhor engajamento."}
                ]
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Erro ao processar resposta do Gemini: {str(e)}")
            gemini_comments = [
                {"comment": f"Duração {duration:.2f}s está {'adequada' if 300 <= duration <= 3600 else 'fora do ideal'}.", "suggestion": "Ajuste a duração para 5-60 minutos."},
                {"comment": f"Clipping {clipping:.4f} {'excede' if clipping > 0.1 else 'está dentro do'} limite.", "suggestion": "Reduza o ganho para evitar clipping."},
                {"comment": f"Silêncio {silence_ratio:.4f} {'é aceitável' if silence_ratio < 0.3 else 'é excessivo'}.", "suggestion": "Minimize pausas longas."},
                {"comment": f"Volume RMS {rms:.4f} {'é adequado' if 0.01 < rms < 0.1 else 'está fora do ideal'}.", "suggestion": "Ajuste o volume para clareza."},
                {"comment": f"Pitch {pitch:.2f} Hz {'é apropriado' if 100 < pitch < 300 else 'está fora do ideal'}.", "suggestion": "Module a voz para melhor engajamento."}
            ]

        # Combinar scores com comentários do Gemini
        for i, comment_data in enumerate(gemini_comments):
            comments[i]["comment"] = comment_data.get("comment", "Não avaliado")
            comments[i]["suggestion"] = comment_data.get("suggestion", "Verifique os dados de áudio.")

        quality_score = sum(c["score"] for c in comments) / len(comments) * 100
        result = {"quality_score": round(quality_score, 2), "comments": comments}
        logger.info(f"Avaliação de áudio concluída: quality_score={result['quality_score']}")
        return result
    except Exception as e:
        logger.error(f"Erro ao avaliar qualidade do áudio: {str(e)}")
        raise