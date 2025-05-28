import librosa
import numpy as np
import json
from datetime import datetime

def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        return {"error": f"Erro ao carregar o áudio: {e}"}

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)
    std_rms = np.std(rms)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    avg_pitch = float(np.mean(pitch_values)) if pitch_values.size > 0 else 0.0
    std_pitch = float(np.std(pitch_values)) if pitch_values.size > 0 else 0.0

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    intervals = librosa.effects.split(y, top_db=20)
    silences = []
    for i in range(1, len(intervals)):
        silence_dur = (intervals[i][0] - intervals[i-1][1]) / sr
        silences.append(silence_dur)
    avg_silence = np.mean(silences) if silences else 0.0

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    avg_zcr = np.mean(zcr)
    std_zcr = np.std(zcr)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_spectral_centroid = np.mean(spectral_centroid)
    std_spectral_centroid = np.std(spectral_centroid)

    # Estimativa simples de clipping: contagem de amostras com valor próximo ao máximo
    clipping_threshold = 0.99 * np.max(np.abs(y))
    clipping_count = int(np.sum(np.abs(y) >= clipping_threshold))

    features = {
        "duration_sec": float(duration),
        "avg_rms": float(avg_rms),
        "std_rms": float(std_rms),
        "avg_pitch": avg_pitch,
        "std_pitch": std_pitch,
        "tempo_bpm": float(tempo) if hasattr(tempo, "__float__") else float(tempo[0]),
        "avg_silence_sec": float(avg_silence),
        "avg_zcr": float(avg_zcr),
        "std_zcr": float(std_zcr),
        "avg_spectral_centroid": float(avg_spectral_centroid),
        "std_spectral_centroid": float(std_spectral_centroid),
        "clipping_count": clipping_count,
        "sample_rate": sr
    }

    return features

def evaluate_quality(features):
    if "error" in features:
        return {"quality_score": 0, "comments": [features["error"]], "passed": False}

    comments = []
    score = 100

    # Duração
    if features["duration_sec"] < 30:
        comments.append("Áudio muito curto (menos de 30 segundos).")
        score -= 20
    else:
        comments.append("Duração adequada.")

    # Clipping
    if features["clipping_count"] > 100:
        comments.append(f"Muitos picos de clipping detectados ({features['clipping_count']} samples).")
        score -= 30

    # Silêncios longos
    if features["avg_silence_sec"] > 5:
        comments.append(f"Pausas longas frequentes (média {features['avg_silence_sec']:.2f} segundos).")
        score -= 15

    # RMS baixo
    if features["avg_rms"] < 0.01:
        comments.append(f"Volume médio muito baixo (RMS médio {features['avg_rms']:.4f}).")
        score -= 15

    # Pitch instável
    if features["std_pitch"] > 50:
        comments.append("Variação alta no pitch, voz instável.")
        score -= 10

    if score < 0:
        score = 0
    elif score > 100:
        score = 100

    passed = score >= 60
    if passed:
        comments.append("Áudio aprovado com qualidade satisfatória.")
    else:
        comments.append("Áudio reprovado devido a problemas detectados.")

    return {
        "quality_score": score,
        "comments": comments,
        "passed": passed
    }

def save_results(features, evaluation, base_filename="audio_quality_report"):
    final_result = {
        "features": features,
        "evaluation": evaluation
    }

    json_path = f"{base_filename}.json"
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(final_result, f_json, indent=4, ensure_ascii=False)

    txt_path = f"{base_filename}.txt"
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write("Relatório de Análise da Qualidade do Áudio\n")
        f_txt.write(f"Arquivo: {features.get('audio_path', base_filename)}\n")
        f_txt.write(f"Duração (segundos): {features.get('duration_sec', 0):.2f}\n")
        f_txt.write(f"Taxa de amostragem: {features.get('sample_rate', 'N/A')} Hz\n")
        f_txt.write(f"Score geral de qualidade: {evaluation['quality_score']} / 100\n")
        f_txt.write(f"Aprovação: {'APROVADO' if evaluation['passed'] else 'REPROVADO'}\n\n")
        f_txt.write("Comentários detalhados:\n")
        for comment in evaluation["comments"]:
            f_txt.write(f"- {comment}\n")

def analyze_audio_quality(audio_path):
    features = extract_audio_features(audio_path)
    if "error" in features:
        return features  # erro já formatado

    evaluation = evaluate_quality(features)

    # Para salvar no relatório, insere caminho do arquivo nas features para referência
    features["audio_path"] = audio_path

    # Gerar timestamp para salvar com nome único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"audio_quality_report_{timestamp}"

    save_results(features, evaluation, base_filename)

    return {
        "message": "Análise concluída",
        "json_report": f"{base_filename}.json",
        "txt_report": f"{base_filename}.txt"
    }

if __name__ == "__main__":
    audio_path = "audio_processing/audio.wav"  # ajuste para seu arquivo real
    result = analyze_audio_quality(audio_path)
    if "error" in result:
        print("Erro:", result["error"])
    else:
        print("Análise concluída!")
        print(f"Relatórios gerados:\n- {result['json_report']}\n- {result['txt_report']}")
