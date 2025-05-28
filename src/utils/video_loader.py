from moviepy.editor import VideoFileClip
import os

def load_and_extract_features(video_path, audio_output_path=None):
    try:
        clip = VideoFileClip(video_path)
        features = {
            "duration_sec": float(clip.duration),
            "fps": clip.fps,
            "resolution": f"{clip.size[0]}x{clip.size[1]}"
        }
        if audio_output_path and clip.audio:
            os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
            clip.audio.write_audiofile(audio_output_path)
            features["audio_path"] = audio_output_path
        clip.close()
        return features
    except Exception as e:
        raise ValueError(f"Erro ao processar vídeo: {e}")