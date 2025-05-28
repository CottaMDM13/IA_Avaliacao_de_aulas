import cv2
import mediapipe as mp
import json
import os
import numpy as np
from datetime import datetime

def analyze_facial_features(video_path):
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)

    eye_positions = []
    nose_positions = []
    mouth_openness = []
    brow_distances = []
    frames_with_face = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            frames_with_face += 1
            landmarks = results.multi_face_landmarks[0].landmark

            # Olhos (média dos olhos esquerdo e direito)
            eye_y = (landmarks[33].y + landmarks[263].y) / 2
            eye_positions.append(eye_y)

            # Nariz (ponta)
            nose_y = landmarks[1].y
            nose_positions.append(nose_y)

            # Boca (abertura)
            mouth_open = abs(landmarks[13].y - landmarks[14].y)
            mouth_openness.append(mouth_open)

            # Sobrancelhas (distância entre topo da sobrancelha e olho)
            brow_left = abs(landmarks[70].y - landmarks[33].y)
            brow_right = abs(landmarks[300].y - landmarks[263].y)
            brow_distances.append((brow_left + brow_right) / 2)

    cap.release()

    if total_frames == 0:
        raise ValueError("Nenhum frame foi lido do vídeo.")

    features = {
        "face_visibility_ratio": round(frames_with_face / total_frames, 4),
        "eye_position_stability": round(np.std(eye_positions), 4) if eye_positions else 0,
        "head_movement_index": round(np.std(nose_positions), 4) if nose_positions else 0,
        "expression_variance": round(np.std(mouth_openness + brow_distances), 4) if mouth_openness and brow_distances else 0
    }

    return features

def evaluate_facial_features(features):
    comments = {}
    score_total = 0
    score_possible = 0

    def score_rule(value, threshold, higher_is_better=True, label=""):
        passed = (value >= threshold) if higher_is_better else (value <= threshold)
        score = 1 if passed else 0
        explanation = f"{label} {'adequado' if passed else 'insuficiente'} (valor: {value}, esperado: {'>=' if higher_is_better else '<='} {threshold})"
        return score, explanation

    rules = [
        ("face_visibility_ratio", 0.8, True, "Presença do rosto"),
        ("eye_position_stability", 0.02, False, "Contato visual"),
        ("head_movement_index", 0.01, True, "Movimento da cabeça"),
        ("expression_variance", 0.005, True, "Expressividade facial")
    ]

    for key, threshold, better, label in rules:
        score, comment = score_rule(features[key], threshold, better, label)
        comments[key] = {"score": score, "comment": comment}
        score_total += score
        score_possible += 1

    overall_score = round(score_total / score_possible, 2)

    return {
        "features": features,
        "comments": comments,
        "overall_score": overall_score
    }

def save_report(data, base_filename="video_facial_analysis"):
    os.makedirs("video_analysis", exist_ok=True)
    json_path = f"video_analysis/{base_filename}.json"
    txt_path = f"video_analysis/{base_filename}.txt"

    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(data, f_json, indent=4, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write("Relatório de Expressões Faciais\n")
        f_txt.write(f"Avaliação geral: {data['overall_score'] * 100:.0f}%\n\n")
        for k, v in data["comments"].items():
            f_txt.write(f"- {k}: {v['comment']} (score: {v['score']})\n")
        f_txt.write("\n")
        if data["overall_score"] < 0.5:
            f_txt.write("Recomendação: Baixa expressividade facial. Considere ser mais expressivo e manter contato visual com a câmera.\n")
        else:
            f_txt.write("Recomendação: Expressividade facial e contato visual adequados.\n")

if __name__ == "__main__":
    video_path = "input_videos/aula_exemplo.mp4"
    features = analyze_facial_features(video_path)
    results = evaluate_facial_features(features)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_report(results, f"facial_analysis_{timestamp}")
    print("Análise facial concluída e relatórios salvos.")
