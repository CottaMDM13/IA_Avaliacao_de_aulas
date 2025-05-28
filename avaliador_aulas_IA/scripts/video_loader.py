from moviepy.editor import VideoFileClip
import os

video_path = "input_videos/aula_exemplo.mp4"
output_audio_path = "audio_processing/audio.wav"

try:
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError("O vídeo não possui faixa de áudio.")
    
    os.makedirs("audio_processing", exist_ok=True)
    clip.audio.write_audiofile(output_audio_path)
    print(f"Áudio extraído com sucesso: {output_audio_path}")
except Exception as e:
    print(f"Erro ao extrair áudio: {e}")
