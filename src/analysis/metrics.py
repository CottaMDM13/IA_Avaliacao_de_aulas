import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_overall_score(video_results, audio_results, text_results, config):
    try:
        logger.info("Calculando métricas gerais")
        weights = {"video": 0.2, "audio": 0.3, "text_didactic": 0.5}  # Pesos fornecidos, soma = 1.0
        
        # Verificar métricas de vídeo
        video_comments = video_results.get("comments", {})
        video_score = video_results.get("overall_score", 0)
        if not video_comments:
            logger.warning("Nenhuma métrica de comentário encontrada em video_results")
        
        # Verificar métricas de áudio
        audio_comments = audio_results.get("comments", [])
        audio_score = audio_results.get("quality_score", 0)
        if not audio_comments:
            logger.warning("Nenhuma métrica de comentário encontrada em audio_results")
        
        # Calcular score médio de vídeo com verificação de existência
        feedback = []
        for aspect, key in [
            ("Vídeo - Postura", "posture"),
            ("Vídeo - Gestos", "gestures"),
            ("Vídeo - Visibilidade do Rosto", "face_visibility"),
            ("Vídeo - Contato Visual", "eye_contact"),
        ]:
            comment_data = video_comments.get(key, {})
            score = comment_data.get("score", 0)
            comment = comment_data.get("comment", "Não avaliado devido à ausência de dados")
            suggestion = comment_data.get("suggestion", "Verifique se o vídeo foi processado corretamente.")
            feedback.append({
                "aspect": aspect,
                "score": score,
                "comment": comment,
                "suggestion": suggestion
            })

        # Adicionar feedback para métricas de áudio
        audio_aspects = ["Duração", "Clipping", "Silêncio", "Volume", "Pitch"]
        for i, aspect_name in enumerate(audio_aspects):
            comment_data = audio_comments[i] if i < len(audio_comments) else {}  # Verificar se índice é válido
            score = comment_data.get("score", 0)
            comment = comment_data.get("comment", f"{aspect_name} não avaliado devido a dados insuficientes")
            suggestion = comment_data.get("suggestion", "Verifique se o áudio foi processado corretamente")
            feedback.append({
                "aspect": f"Áudio - {aspect_name}",
                "score": score,
                "comment": comment,
                "suggestion": suggestion
            })

        # Adicionar feedback para texto didático
        feedback.append({
            "aspect": "Texto - Didática",
            "score": text_results.get("didactic_score", 0) / 10,
            "comment": text_results.get("didactic_justification", "N/A"),
            "suggestion": (
                "Estruture melhor o conteúdo com exemplos práticos e perguntas para engajar os alunos." if text_results.get("didactic_score", 0) / 10 < 0.9
                else "Continue estruturando o conteúdo de forma clara e envolvente."
            )
        })

        # Calcular média ponderada para texto geral
        text_general_score = (
            weights["video"] * video_score +
            weights["audio"] * audio_score / 100 +
            weights["text_didactic"] * text_results.get("didactic_score", 0) / 10
        )
        # Agregar comentários para texto geral
        video_summary = "Vídeo: " + ", ".join(
            f"{k}: {v['comment']}" for k, v in video_comments.items() if v.get("comment")
        ) if video_comments else "Vídeo: Não avaliado."
        audio_summary = "Áudio: " + ", ".join(
            f"{audio_aspects[i]}: {c.get('comment', 'N/A')}" for i, c in enumerate(audio_comments)
        ) if audio_comments else "Áudio: Não avaliado."
        didactic_summary = f"Didática: {text_results.get('didactic_justification', 'N/A')}"
        general_comment = f"{video_summary} {audio_summary} {didactic_summary}"
        general_suggestion = (
            "Integre melhor os elementos de vídeo, áudio e didática para uma aula mais coesa." if text_general_score < 0.9
            else "Continue integrando bem os elementos da aula."
        )
        feedback.append({
            "aspect": "Texto - Avaliação Geral",
            "score": round(text_general_score, 2),
            "comment": general_comment,
            "suggestion": general_suggestion
        })

        # Calcular score geral
        overall_score = (
            weights["video"] * video_score +
            weights["audio"] * audio_score / 100 +
            weights["text_didactic"] * text_results.get("didactic_score", 0) / 10
        )
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