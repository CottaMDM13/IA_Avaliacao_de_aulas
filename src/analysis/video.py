import cv2
import mediapipe as mp
import numpy as np
import os
import logging
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import json
import requests

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
            # Verificar conexão com a API
            try:
                response = requests.get("https://generativelanguage.googleapis.com/v1beta/models?key=" + api_key, timeout=5)
                if response.status_code != 200:
                    logger.error(f"Falha na conexão com a API Gemini: {response.text}")
                    raise ValueError("Falha na conexão com a API Gemini")
            except requests.RequestException as e:
                logger.error(f"Erro de rede ao conectar com a API Gemini: {str(e)}")
                raise
            _model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Modelo Gemini inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo Gemini: {str(e)}")
            raise
    return _model

def analyze_video(video_path):
    try:
        logger.info(f"Analisando vídeo: {video_path}")
        mp_pose = mp.solutions.pose
        mp_face = mp.solutions.face_mesh
        pose = mp_pose.Pose()
        face_mesh = mp_face.FaceMesh(static_image_mode=False)
        cap = cv2.VideoCapture(video_path)

        posture_scores, movement_magnitudes, hand_movements = [], [], 0
        eye_positions, nose_positions, mouth_openness, brow_distances = [], [], [], []
        total_frames = 0
        prev_left_wrist, prev_right_wrist = None, None
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Processar a cada 5 frames para otimizar
            if frame_count % 5 != 0:
                continue
            total_frames += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Análise de gestos
            pose_results = pose.process(image_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                posture_score = max(0, 1 - abs(left_shoulder.y - right_shoulder.y) * 5)
                posture_scores.append(posture_score)

                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                if prev_left_wrist and prev_right_wrist:
                    lw_movement = ((left_wrist.x - prev_left_wrist.x) ** 2 + (left_wrist.y - prev_left_wrist.y) ** 2) ** 0.5
                    rw_movement = ((right_wrist.x - prev_right_wrist.x) ** 2 + (right_wrist.y - prev_right_wrist.y) ** 2) ** 0.5
                    movement_magnitudes.append((lw_movement + rw_movement) / 2)
                    if lw_movement > 0.01 or rw_movement > 0.01:
                        hand_movements += 1
                prev_left_wrist, prev_right_wrist = left_wrist, right_wrist

            # Análise de expressões faciais
            face_results = face_mesh.process(image_rgb)
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                eye_y = (landmarks[33].y + landmarks[263].y) / 2
                eye_positions.append(eye_y)
                nose_y = landmarks[1].y
                nose_positions.append(nose_y)
                mouth_open = abs(landmarks[13].y - landmarks[14].y)
                mouth_openness.append(mouth_open)
                brow_left = abs(landmarks[70].y - landmarks[33].y)
                brow_right = abs(landmarks[300].y - landmarks[263].y)
                brow_distances.append((brow_left + brow_right) / 2)

        cap.release()
        pose.close()
        face_mesh.close()

        gesture_metrics = {
            "avg_posture_score": round(sum(posture_scores) / len(posture_scores), 4) if posture_scores else 0,
            "avg_hand_movement": round(sum(movement_magnitudes) / len(movement_magnitudes), 4) if movement_magnitudes else 0,
            "total_hand_movements": hand_movements,
            "total_frames": total_frames
        }
        facial_metrics = {
            "eye_position_stability": round(np.std(eye_positions), 4) if eye_positions else 0,
            "head_movement_index": round(np.std(nose_positions), 4) if nose_positions else 0,
            "expression_variance": round(np.std(mouth_openness + brow_distances), 4) if mouth_openness and brow_distances else 0
        }

        result = {"gestures": gesture_metrics, "facial_expressions": facial_metrics}
        logger.info(f"Análise de vídeo concluída: {video_path}")
        return result
    except Exception as e:
        logger.error(f"Erro ao analisar vídeo {video_path}: {str(e)}")
        raise

def evaluate_video_quality(metrics, config):
    try:
        logger.debug("Avaliando qualidade do vídeo")
        model = get_gemini_model()
        comments = {}

        # Calcular escores mantendo a lógica existente
        posture_score = metrics["gestures"]["avg_posture_score"]
        if posture_score > 0.9:
            score = 0.95
        elif posture_score > 0.85:
            score = 0.85
        elif posture_score > 0.7:
            score = 0.7
        elif posture_score > 0.5:
            score = 0.5
        else:
            score = 0.3
        comments["posture"] = {"score": score}

        hand_movement = metrics["gestures"]["avg_hand_movement"]
        if hand_movement > 0.015:
            score = 0.95
        elif hand_movement > 0.01:
            score = 0.85
        elif hand_movement > 0.005:
            score = 0.6
        elif hand_movement > 0.002:
            score = 0.4
        else:
            score = 0.2
        comments["gestures"] = {"score": score}

        eye_stability = metrics["facial_expressions"]["eye_position_stability"]
        if eye_stability < 0.015:
            score = 0.95
        elif eye_stability < 0.02:
            score = 0.85
        elif eye_stability < 0.03:
            score = 0.6
        else:
            score = 0.3
        comments["eye_contact"] = {"score": score}

        # Gerar comentários via Gemini com prompt ajustado
        prompt = (
            "Você é um especialista em análise de vídeo educacional. Com base nas métricas de vídeo fornecidas, gere comentários e sugestões para cada aspecto (postura, gestos, contato visual) "
            "em um tom neutro, educacional e objetivo. Forneça um resumo narrativo (máx. 50 palavras) e uma sugestão de melhoria para cada aspecto. "
            "Retorne um JSON estruturado com 'comment' e 'suggestion' para cada aspecto, mesmo que os valores sejam genéricos. "
            "Formato esperado: "
            "{\"posture\": {\"comment\": \"string\", \"suggestion\": \"string\"}, "
            "\"gestures\": {\"comment\": \"string\", \"suggestion\": \"string\"}, "
            "\"eye_contact\": {\"comment\": \"string\", \"suggestion\": \"string\"}} "
            "Métricas fornecidas:\n"
            f"Postura: avg_posture_score={metrics['gestures']['avg_posture_score']:.4f} (ideal: >0.9)\n"
            f"Gestos: avg_hand_movement={metrics['gestures']['avg_hand_movement']:.4f}, total_hand_movements={metrics['gestures']['total_hand_movements']} (ideal: >0.015)\n"
            f"Contato Visual: eye_position_stability={metrics['facial_expressions']['eye_position_stability']:.4f} (ideal: <0.015)\n"
            "Exemplo: "
            "{\"posture\": {\"comment\": \"Postura estável com alinhamento adequado.\", \"suggestion\": \"Mantenha a postura ereta.\"}, "
            "\"gestures\": {\"comment\": \"Gestos expressivos reforçam a comunicação.\", \"suggestion\": \"Continue usando gestos naturais.\"}}"
        )
        try:
            response = model.generate_content(prompt)
            logger.debug(f"Resposta bruta do Gemini: {response.text}")
            gemini_comments = json.loads(response.text.strip("```json\n").strip("\n```"))
            logger.debug(f"Resposta parseada do Gemini: {gemini_comments}")
            # Validar a estrutura da resposta
            required_fields = ["posture", "gestures", "eye_contact"]
            for field in required_fields:
                if field not in gemini_comments or "comment" not in gemini_comments[field] or "suggestion" not in gemini_comments[field]:
                    logger.warning(f"Campo {field} ausente ou inválido na resposta do Gemini")
                    gemini_comments[field] = {
                        "comment": f"{field} não avaliado devido a resposta inválida.",
                        "suggestion": "Verifique os dados do vídeo."
                    }
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Erro ao processar resposta do Gemini: {str(e)}")
            gemini_comments = {
                "posture": {
                    "comment": f"Postura com pontuação {posture_score:.4f} está {'excelente' if posture_score > 0.9 else 'adequada' if posture_score > 0.7 else 'abaixo do ideal'}.",
                    "suggestion": "Mantenha uma postura ereta e equilibrada."
                },
                "gestures": {
                    "comment": f"Gestos com movimento médio {hand_movement:.4f} {'são expressivos' if hand_movement > 0.015 else 'são moderados' if hand_movement > 0.005 else 'são limitados'}.",
                    "suggestion": "Incorpore gestos naturais para reforçar a comunicação."
                },
                "eye_contact": {
                    "comment": f"Estabilidade do contato visual {eye_stability:.4f} {'é excelente' if eye_stability < 0.015 else 'precisa de ajustes'}.",
                    "suggestion": "Mantenha o olhar direcionado para a câmera."
                }
            }

        # Atribuir comentários do Gemini
        for aspect in ["posture", "gestures", "eye_contact"]:
            comment_data = gemini_comments.get(aspect, {})
            comments[aspect]["comment"] = comment_data.get("comment", f"{aspect} não avaliado.")
            comments[aspect]["suggestion"] = comment_data.get("suggestion", "Verifique os dados do vídeo.")

        overall_score = sum(c["score"] for c in comments.values()) / len(comments)
        result = {"metrics": metrics, "comments": comments, "overall_score": round(overall_score, 2)}
        logger.info(f"Avaliação de vídeo concluída: overall_score={result['overall_score']}")
        return result
    except Exception as e:
        logger.error(f"Erro ao avaliar qualidade do vídeo: {str(e)}")
        raise