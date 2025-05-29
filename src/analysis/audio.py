import librosa
import numpy as np
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_audio_features(audio_path):
    try:
        logger.debug(f"Extraindo features de áudio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None)
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

        result = {
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
        logger.info(f"Features de áudio extraídas: {audio_path}")
        return result
    except Exception as e:
        logger.error(f"Erro ao extrair features de áudio {audio_path}: {str(e)}")
        return {"error": f"Erro ao carregar áudio: {e}"}

def evaluate_audio_quality(features, config):
    try:
        logger.debug("Avaliando qualidade do áudio")
        if "error" in features:
            result = {"quality_score": 0, "comments": [{"score": 0, "comment": features["error"]}], "passed": False}
            logger.warning(f"Erro na avaliação de áudio: {features['error']}")
            return result

        score, comments = 100, []
        
        # Duração
        if features["duration_sec"] < 30:
            comments.append({"score": 0.3, "comment": "Áudio muito curto, menos de 30 segundos."})
            score -= 20
        elif features["duration_sec"] < 60:
            comments.append({"score": 0.6, "comment": "Áudio curto, menos de 1 minuto."})
            score -= 10
        else:
            comments.append({"score": 0.9, "comment": "Duração adequada para análise."})

        # Clipping
        if features["clipping_count"] > 200:
            comments.append({"score": 0.2, "comment": f"Clipping excessivo ({features['clipping_count']} samples)."})
            score -= 40
        elif features["clipping_count"] > 100:
            comments.append({"score": 0.5, "comment": f"Muitos picos de clipping ({features['clipping_count']} samples)."})
            score -= 20
        else:
            comments.append({"score": 0.9, "comment": "Sem clipping significativo."})

        # Silêncio
        if features["avg_silence_sec"] > 7:
            comments.append({"score": 0.3, "comment": f"Pausas muito longas (média {features['avg_silence_sec']:.2f}s)."})
            score -= 20
        elif features["avg_silence_sec"] > 5:
            comments.append({"score": 0.5, "comment": f"Pausas longas frequentes (média {features['avg_silence_sec']:.2f}s)."})
            score -= 10
        else:
            comments.append({"score": 0.9, "comment": "Pausas adequadas."})

        # Volume (RMS)
        if features["avg_rms"] < 0.005:
            comments.append({"score": 0.3, "comment": f"Volume muito baixo (RMS médio {features['avg_rms']:.4f})."})
            score -= 20
        elif features["avg_rms"] < 0.01:
            comments.append({"score": 0.6, "comment": f"Volume baixo (RMS médio {features['avg_rms']:.4f})."})
            score -= 10
        else:
            comments.append({"score": 0.9, "comment": "Volume adequado."})

        # Pitch
        if features["std_pitch"] > 70:
            comments.append({"score": 0.3, "comment": "Variação muito alta no pitch, voz instável."})
            score -= 20
        elif features["std_pitch"] > 50:
            comments.append({"score": 0.5, "comment": "Variação alta no pitch, voz instável."})
            score -= 10
        elif features["std_pitch"] < 20:
            comments.append({"score": 0.6, "comment": "Voz monótona, pouca variação no pitch."})
            score -= 5
        else:
            comments.append({"score": 0.9, "comment": "Variação de pitch adequada."})

        score = max(0, min(100, score))
        comments.append({"score": score/100, "comment": "Áudio aprovado." if score >= config["thresholds"]["audio_score"] else "Áudio reprovado."})
        result = {"quality_score": score, "comments": comments, "passed": score >= config["thresholds"]["audio_score"]}
        logger.info(f"Avaliação de áudio concluída: quality_score={result['quality_score']}")
        return result
    except Exception as e:
        logger.error(f"Erro ao avaliar qualidade do áudio: {str(e)}")
        raise
