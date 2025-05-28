from moviepy.editor import VideoFileClip
import json
import os

def extract_video_features(video_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration  # duração em segundos
    fps = clip.fps
    width, height = clip.size
    resolution = f"{width}x{height}"

    features = {
        "duration_sec": float(duration),
        "fps": fps,
        "resolution": resolution
    }

    clip.close()
    return features

if __name__ == "__main__":
    video_path = "input_videos/aula_exemplo.mp4"  # ajuste se estiver em outro local
    output_path = "video_analysis/video_features.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    feats = extract_video_features(video_path)
    print("Features extraídas do vídeo:")
    for k, v in feats.items():
        print(f"{k}: {v}")

    with open(output_path, "w") as f:
        json.dump(feats, f, indent=4)
