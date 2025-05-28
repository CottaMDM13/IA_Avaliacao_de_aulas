def calculate_overall_score(video_results, audio_results, text_results, config):
    weights = {"video": 0.3, "audio": 0.3, "text": 0.4}
    overall_score = (
        weights["video"] * video_results["overall_score"] +
        weights["audio"] * audio_results["evaluation"]["quality_score"] / 100 +
        weights["text"] * text_results.get("overall_score", 0) / 10
    )
    feedback = [
        f"Video: {video_results['comments']['posture']['comment']}",
        f"Audio: {audio_results['evaluation']['comments'][0]}",
        f"Texto: {text_results.get('justification', 'N/A')}"
    ]
    return {
        "overall_score": round(overall_score, 2),
        "approved": overall_score >= config["thresholds"]["overall_score"],
        "feedback": feedback
    }