import os
import json
import librosa
import numpy as np

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)
    std_rms = np.std(rms)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    avg_pitch = np.mean(pitch_values) if pitch_values.size > 0 else 0
    std_pitch = np.std(pitch_values) if pitch_values.size > 0 else 0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    intervals = librosa.effects.split(y, top_db=20)
    silences = []
    for i in range(1, len(intervals)):
        silence_dur = (intervals[i][0] - intervals[i-1][1]) / sr
        silences.append(silence_dur)
    avg_silence = np.mean(silences) if silences else 0
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    avg_zcr = np.mean(zcr)
    std_zcr = np.std(zcr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_spectral_centroid = np.mean(spectral_centroid)
    std_spectral_centroid = np.std(spectral_centroid)

    # Dynamic range (max - min amplitude)
    dynamic_range = float(np.max(y) - np.min(y))

    # Clipping detection: quantos samples chegam perto de 1.0/-1.0 (limite)
    clipping_thresh = 0.99
    clipping_count = int(np.sum(np.abs(y) > clipping_thresh))

    # MFCCs (13 coeficientes padrão)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1).tolist()
    mfcc_stds = np.std(mfcc, axis=1).tolist()

    # SNR estimado simples (média RMS / RMS do silêncio)
    # Para isso, pegamos os intervalos silenciosos e calculamos rms deles
    silence_segments = [y[intervals[i-1][1]:intervals[i][0]] for i in range(1, len(intervals))]
    silence_rms = np.mean([np.sqrt(np.mean(seg**2)) for seg in silence_segments]) if silence_segments else 0
    snr = (avg_rms / silence_rms) if silence_rms > 0 else float('inf')

    features = {
        "duration_sec": float(duration),
        "avg_rms": float(avg_rms),
        "std_rms": float(std_rms),
        "avg_pitch": float(avg_pitch),
        "std_pitch": float(std_pitch),
        "tempo_bpm": float(tempo) if not hasattr(tempo, "__getitem__") else float(tempo[0]),
        "avg_silence_sec": float(avg_silence),
        "avg_zcr": float(avg_zcr),
        "std_zcr": float(std_zcr),
        "avg_spectral_centroid": float(avg_spectral_centroid),
        "std_spectral_centroid": float(std_spectral_centroid),
        "dynamic_range": dynamic_range,
        "clipping_count": clipping_count,
        "snr_estimated": float(snr),
        "mfcc_means": mfcc_means,
        "mfcc_stds": mfcc_stds
    }

    return features

if __name__ == "__main__":
    audio_path = "audio_processing/audio.wav"  # ajuste seu caminho
    feats = extract_audio_features(audio_path)

    output_folder = "audio_analysis"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "audio_features.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feats, f, indent=4, ensure_ascii=False)

    print(f"Features salvas em {output_path}")
