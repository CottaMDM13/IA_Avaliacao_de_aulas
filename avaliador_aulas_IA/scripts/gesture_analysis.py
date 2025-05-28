import cv2
import mediapipe as mp
import json
import os
from datetime import datetime


def analyze_gestures(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    posture_scores = []
    movement_magnitudes = []
    hand_movements = 0

    prev_left_wrist = None
    prev_right_wrist = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            distance = abs(left_shoulder.y - right_shoulder.y)
            posture_score = max(0, 1 - distance * 5)  # penaliza inclinação
            posture_scores.append(posture_score)

            # Movimento das mãos (dinâmica)
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            if prev_left_wrist and prev_right_wrist:
                lw_movement = ((left_wrist.x - prev_left_wrist.x) ** 2 + (left_wrist.y - prev_left_wrist.y) ** 2) ** 0.5
                rw_movement = ((right_wrist.x - prev_right_wrist.x) ** 2 + (right_wrist.y - prev_right_wrist.y) ** 2) ** 0.5
                movement_magnitudes.append((lw_movement + rw_movement) / 2)

                if lw_movement > 0.01 or rw_movement > 0.01:
                    hand_movements += 1

            prev_left_wrist = left_wrist
            prev_right_wrist = right_wrist

    cap.release()

    avg_posture = sum(posture_scores) / len(posture_scores) if posture_scores else 0
    avg_hand_movement = sum(movement_magnitudes) / len(movement_magnitudes) if movement_magnitudes else 0

    return {
        "avg_posture_score": round(avg_posture, 4),
        "avg_hand_movement": round(avg_hand_movement, 4),
        "total_hand_movements": hand_movements,
        "total_frames": total_frames
    }


def evaluate_gesture_quality(metrics):
    comments = {}

    if metrics["avg_posture_score"] > 0.85:
        comments["posture"] = {
            "score": 1,
            "comment": "Postura boa e constante."
        }
    elif metrics["avg_posture_score"] > 0.6:
        comments["posture"] = {
            "score": 0.5,
            "comment": "Postura aceitável, mas pode melhorar."
        }
    else:
        comments["posture"] = {
            "score": 0,
            "comment": "Postura inadequada durante a apresentação."
        }

    if metrics["avg_hand_movement"] > 0.01:
        comments["gestures"] = {
            "score": 1,
            "comment": "Boa utilização de gestos com as mãos."
        }
    elif metrics["avg_hand_movement"] > 0.005:
        comments["gestures"] = {
            "score": 0.5,
            "comment": "Pouca movimentação das mãos, poderia ser mais expressivo."
        }
    else:
        comments["gestures"] = {
            "score": 0,
            "comment": "Pouco ou nenhum gesto detectado."
        }

    overall_score = sum(c["score"] for c in comments.values()) / len(comments)

    return {
        "metrics": metrics,
        "comments": comments,
        "overall_score": round(overall_score, 2)
    }


def save_gesture_report(results, base_path="video_analysis/gesture_report"):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # JSON
    with open(base_path + ".json", "w") as f_json:
        json.dump(results, f_json, indent=4)

    # TXT
    with open(base_path + ".txt", "w", encoding="utf-8") as f_txt:
        f_txt.write("Relatório de Análise de Gestos Corporais\n")
        f_txt.write("--------------------------------------\n")
        f_txt.write(f"Total de quadros analisados: {results['metrics']['total_frames']}\n")
        f_txt.write(f"Pontuação geral: {results['overall_score'] * 100:.0f}%\n\n")

        for key, data in results["comments"].items():
            f_txt.write(f"- {key.upper()}\n  Comentário: {data['comment']}\n  Score: {data['score']}\n\n")

        if results['overall_score'] >= 0.7:
            f_txt.write("Conclusão: Apresentação com linguagem corporal adequada.\n")
        else:
            f_txt.write("Conclusão: Linguagem corporal pode ser aprimorada.\n")


if __name__ == "__main__":
    video_file = "input_videos/aula_exemplo.mp4"
    metrics = analyze_gestures(video_file)
    results = evaluate_gesture_quality(metrics)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"video_analysis/gesture_report_{timestamp}"
    save_gesture_report(results, base_name)
    print(f"Relatório gerado: {base_name}.json e .txt")
