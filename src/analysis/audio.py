import librosa
import numpy as np
import os
from datetime import datetime

def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        return {"error": f"Erro ao carregar áudio: {e}"}

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    intervals = librosa.effects.split(y, top_db=20)
    silences = [(intervals[i][0] - intervals[i-1][1]) / sr for i in range(1, len(intervals))]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    clipping_threshold = 0.99 * np.max(np.abs(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    silence_segments = [y[intervals[i-1][1]:intervals[i][0]] for i in range(1, len(intervals))]
    silence_rms = np.mean([np.sqrt(np.mean(seg**2)) for seg in silence_segments]) if silence_segments else 0

    return {
        "duration_sec": float(duration),
        "avg_rms": float(np.mean(rms)),
        "std_rms": float(np.std(rms)),
        "avg_pitch": float(np.mean(pitch_values)) if pitch_values.size > 0 else 0,
        "std_pitch": float(np.std(pitch_values)) if pitch_values.size > 0 else 0,
        "tempo_bpm": float(librosa.beat.beat_track(y=y, sr=sr)[0]),
        "avg_silence_sec": float(np.mean(silences)) if silences else 0,
        "avg_zcr": float(np.mean(zcr)),
        "std_zcr": float(np.std(zcr)),
        "avg_spectral_centroid": float(np.mean(spectral_centroid)),
        "std_spectral_centroid": float(np.std(spectral_centroid)),
        "clipping_count": int(np.sum(np.abs(y) >= clipping_threshold)),
        "snr_estimated": float(np.mean(rms) / silence_rms) if silence_rms > 0 else float('inf'),
        "mfcc_means": np.mean(mfcc, axis=1).tolist(),
        "mfcc_stds": np.std(mfcc, axis=1).tolist(),
        "audio_path": audio_path
    }

def evaluate_audio_quality(features, config):
    if "error" in features:
        return {"quality_score": 0, "comments": [features["error"]], "passed": False}

    score, comments = 100, []
    if features["duration_sec"] < 30:
        comments.append("Áudio muito curto.")
        score -= 20
    if features["clipping_count"] > 100:
        comments.append(f"Muitos picos de clipping ({features['clipping_count']} samples).")
        score -= 30
    if features["avg_silence_sec"] > 5:
        comments.append(f"Pausas longas frequentes (média {features['avg_silence_sec']:.2f}s).")
        score -= 15
    if features["avg_rms"] < 0.01:
        comments.append(f"Volume baixo (RMS médio {features['avg_rms']:.4f}).")
        score -= 15
    if features["std_pitch"] > 50:
        comments.append("Variação alta no pitch, voz instável.")
        score -= 10
    score = max(0, min(100, score))
    comments.append("Áudio aprovado." if score >= config["thresholds"]["audio_score"] else "Áudio reprovado.")
    return {"quality_score": score, "comments": comments, "passed": score >= config["thresholds"]["audio_score"]}