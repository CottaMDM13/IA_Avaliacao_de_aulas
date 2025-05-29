import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_overall_score(video_results, audio_results, text_results, config):
    try:
        weights = {"video": 0.3, "audio": 0.3, "text": 0.4}
        overall_score = (
            weights["video"] * video_results["overall_score"] +
            weights["audio"] * audio_results["quality_score"] / 100 +
            weights["text"] * text_results.get("overall_score", 0) / 10
        )
        feedback = [
            {
                "aspect": "Vídeo - Postura",
                "score": video_results["comments"]["posture"]["score"],
                "comment": video_results["comments"]["posture"]["comment"],
                "suggestion": (
                    "Mantenha os ombros alinhados e evite inclinações laterais." if video_results["comments"]["posture"]["score"] < 0.9
                    else ""
                )
            },
            {
                "aspect": "Vídeo - Gestos",
                "score": video_results["comments"]["gestures"]["score"],
                "comment": video_results["comments"]["gestures"]["comment"],
                "suggestion": (
                    "Use gestos mais amplos e intencionais para enfatizar pontos-chave." if video_results["comments"]["gestures"]["score"] < 0.9
                    else ""
                )
            },
            {
                "aspect": "Vídeo - Visibilidade do Rosto",
                "score": video_results["comments"]["face_visibility"]["score"],
                "comment": video_results["comments"]["face_visibility"]["comment"],
                "suggestion": (
                    "Ajuste a câmera para manter o rosto visível durante toda a aula." if video_results["comments"]["face_visibility"]["score"] < 0.9
                    else ""
                )
            },
            {
                "aspect": "Vídeo - Contato Visual",
                "score": video_results["comments"]["eye_contact"]["score"],
                "comment": video_results["comments"]["eye_contact"]["comment"],
                "suggestion": (
                    "Olhe diretamente para a câmera para simular contato visual com os alunos." if video_results["comments"]["eye_contact"]["score"] < 0.9
                    else ""
                )
            },
            {
                "aspect": "Áudio - Duração",
                "score": audio_results["comments"][0]["score"],
                "comment": audio_results["comments"][0]["comment"],
                "suggestion": (
                    "Estenda a duração da aula para cobrir o conteúdo de forma mais completa." if audio_results["comments"][0]["score"] < 0.9
                    else ""
                )
            },
            {
                "aspect": "Áudio - Clipping",
                "score": audio_results["comments"][1]["score"],
                "comment": audio_results["comments"][1]["comment"],
                "suggestion": (
                    "Ajuste o ganho do microfone para evitar picos de clipping." if audio_results["comments"][1]["score"] < 0.9
                    else ""
                )
            },
            {
                "aspect": "Áudio - Silêncio",
                "score": audio_results["comments"][2]["score"],
                "comment": audio_results["comments"][2]["comment"],
                "suggestion": (
                    "Reduza pausas longas, mantendo um ritmo fluido na fala." if audio_results["comments"][2]["score"] < 0.9
                    else ""
                )
            },
            {
                "aspect": "Áudio - Volume",
                "score": audio_results["comments"][3]["score"],
                "comment": audio_results["comments"][3]["comment"],
                "suggestion": (
                    "Aumente o volume da voz ou aproxime-se do microfone." if audio_results["comments"][3]["score"] < 0.9
                    else ""
                )
            },
            {
                "aspect": "Áudio - Pitch",
                "score": audio_results["comments"][4]["score"],
                "comment": audio_results["comments"][4]["comment"],
                "suggestion": (
                    "Varie o tom de voz para manter o engajamento, evitando monotonia ou instabilidade." if audio_results["comments"][4]["score"] < 0.9
                    else ""
                )
            },
            {
                "aspect": "Texto - Didática e Engajamento",
                "score": text_results.get("overall_score", 0)/10,
                "comment": text_results.get("justification", "N/A"),
                "suggestion": (
                    "Estruture melhor o conteúdo com exemplos práticos e perguntas para engajar os alunos." if text_results.get("overall_score", 0)/10 < 0.9
                    else ""
                )
            }
        ]
        result = {
            "overall_score": round(overall_score, 2),
            "approved": overall_score >= config["thresholds"]["overall_score"],
            "feedback": feedback
        }
        logger.info(f"Métricas calculadas: overall_score={result['overall_score']}, approved={result['approved']}")
        return result
    except Exception as e:
        logger.error(f"Erro ao calcular métricas: {str(e)}")
        raise
