import cv2
import mediapipe as mp
import numpy as np
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
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
        comments = {}
        posture_score = metrics["gestures"]["avg_posture_score"]
        if posture_score > 0.9:
            comments["posture"] = {"score": 0.95, "comment": "Postura excelente, muito estável."}
        elif posture_score > 0.85:
            comments["posture"] = {"score": 0.85, "comment": "Postura boa e constante."}
        elif posture_score > 0.7:
            comments["posture"] = {"score": 0.7, "comment": "Postura aceitável, mas com leves desvios."}
        elif posture_score > 0.5:
            comments["posture"] = {"score": 0.5, "comment": "Postura irregular, precisa de ajustes."}
        else:
            comments["posture"] = {"score": 0.3, "comment": "Postura inadequada, compromete a apresentação."}

        hand_movement = metrics["gestures"]["avg_hand_movement"]
        if hand_movement > 0.015:
            comments["gestures"] = {"score": 0.95, "comment": "Gestos expressivos, enriquecem a aula."}
        elif hand_movement > 0.01:
            comments["gestures"] = {"score": 0.85, "comment": "Boa utilização de gestos."}
        elif hand_movement > 0.005:
            comments["gestures"] = {"score": 0.6, "comment": "Gestos moderados, podem ser mais dinâmicos."}
        elif hand_movement > 0.002:
            comments["gestures"] = {"score": 0.4, "comment": "Pouca movimentação das mãos."}
        else:
            comments["gestures"] = {"score": 0.2, "comment": "Quase nenhum gesto, aula estática."}

        face_visibility = metrics["facial_expressions"]["face_visibility_ratio"]
        if face_visibility > config["thresholds"]["face_visibility"]:
            comments["face_visibility"] = {"score": 0.9, "comment": "Rosto visível na maior parte do tempo."}
        elif face_visibility > 0.6:
            comments["face_visibility"] = {"score": 0.6, "comment": "Rosto visível em parte do tempo."}
        else:
            comments["face_visibility"] = {"score": 0.3, "comment": "Rosto pouco visível, dificulta engajamento."}

        eye_stability = metrics["facial_expressions"]["eye_position_stability"]
        if eye_stability < 0.015:
            comments["eye_contact"] = {"score": 0.95, "comment": "Excelente contato visual."}
        elif eye_stability < 0.02:
            comments["eye_contact"] = {"score": 0.85, "comment": "Bom contato visual."}
        elif eye_stability < 0.03:
            comments["eye_contact"] = {"score": 0.6, "comment": "Contato visual moderado."}
        else:
            comments["eye_contact"] = {"score": 0.3, "comment": "Contato visual insuficiente."}

        overall_score = sum(c["score"] for c in comments.values()) / len(comments)
        result = {"metrics": metrics, "comments": comments, "overall_score": round(overall_score, 2)}
        logger.info(f"Avaliação de vídeo concluída: overall_score={result['overall_score']}")
        return result
    except Exception as e:
        logger.error(f"Erro ao avaliar qualidade do vídeo: {str(e)}")
        raise
