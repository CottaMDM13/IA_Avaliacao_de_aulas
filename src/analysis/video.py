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
        frames_with_face, total_frames = 0, 0
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
                frames_with_face += 1
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
            "face_visibility_ratio": round(frames_with_face / total_frames, 4) if total_frames else 0,
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

        face_visibility = metrics["facial_expressions"]["face_visibility_ratio"]
        if face_visibility > config["thresholds"]["face_visibility"]:
            score = 0.9
        elif face_visibility > 0.6:
            score = 0.6
        else:
            score = 0.3
        comments["face_visibility"] = {"score": score}

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

        # Gerar comentários via Gemini com prompt mais robusto
        prompt = (
            "Com base nas métricas de vídeo fornecidas, gere comentários e sugestões para cada aspecto (postura, gestos, visibilidade do rosto, contato visual) "
            "em um tom neutro e educacional. Retorne um JSON estruturado com 'comment' e 'suggestion' para cada aspecto, mesmo que os valores sejam genéricos. "
            "Se uma métrica estiver ausente, forneça um comentário e sugestão padrão. "
            "Formato esperado: "
            "{\"posture\": {\"comment\": \"string\", \"suggestion\": \"string\"}, "
            "\"gestures\": {\"comment\": \"string\", \"suggestion\": \"string\"}, "
            "\"face_visibility\": {\"comment\": \"string\", \"suggestion\": \"string\"}, "
            "\"eye_contact\": {\"comment\": \"string\", \"suggestion\": \"string\"}} "
            "Métricas:\n"
            f"Postura: avg_posture_score={metrics['gestures']['avg_posture_score']}\n"
            f"Gestos: avg_hand_movement={metrics['gestures']['avg_hand_movement']}, total_hand_movements={metrics['gestures']['total_hand_movements']}\n"
            f"Visibilidade do Rosto: face_visibility_ratio={metrics['facial_expressions']['face_visibility_ratio']}\n"
            f"Contato Visual: eye_position_stability={metrics['facial_expressions']['eye_position_stability']}\n"
        )
        try:
            response = model.generate_content(prompt)
            logger.debug(f"Resposta bruta do Gemini: {response.text}")
            gemini_comments = json.loads(response.text.strip("```json\n").strip("\n```"))
            logger.debug(f"Resposta parseada do Gemini: {gemini_comments}")
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar resposta do Gemini: {str(e)}")
            gemini_comments = {}
        except Exception as e:
            logger.error(f"Erro ao gerar conteúdo com Gemini: {str(e)}")
            gemini_comments = {}

        # Função para gerar comentários padrão baseados na pontuação
        def get_default_comments(aspect, score):
            if score >= 0.9:
                comment = f"{aspect} está excelente, mantendo um alto padrão."
                suggestion = f"Continue mantendo o excelente desempenho em {aspect.lower()}."
            elif score >= 0.7:
                comment = f"{aspect} está adequado, mas há espaço para melhorias."
                suggestion = f"Considere ajustar {aspect.lower()} para alcançar maior consistência."
            else:
                comment = f"{aspect} apresenta desempenho abaixo do esperado."
                suggestion = f"Recomenda-se revisar {aspect.lower()} para melhorar a qualidade."
            return {"comment": comment, "suggestion": suggestion}

        # Atribuir comentários com verificação de existência
        for aspect, aspect_name in [
            ("posture", "Postura"),
            ("gestures", "Gestos"),
            ("face_visibility", "Visibilidade do Rosto"),
            ("eye_contact", "Contato Visual")
        ]:
            comment_data = gemini_comments.get(aspect, {})
            if comment_data.get("comment") and comment_data.get("suggestion"):
                comments[aspect]["comment"] = comment_data["comment"]
                comments[aspect]["suggestion"] = comment_data["suggestion"]
            else:
                default_comments = get_default_comments(aspect_name, comments[aspect]["score"])
                comments[aspect]["comment"] = default_comments["comment"]
                comments[aspect]["suggestion"] = default_comments["suggestion"]
                logger.warning(f"Usando comentários padrão para {aspect} devido a falha na resposta do Gemini")

        overall_score = sum(c["score"] for c in comments.values()) / len(comments)
        result = {"metrics": metrics, "comments": comments, "overall_score": round(overall_score, 2)}
        logger.info(f"Avaliação de vídeo concluída: overall_score={result['overall_score']}")
        return result
    except Exception as e:
        logger.error(f"Erro ao avaliar qualidade do vídeo: {str(e)}")
        raise