import json
import os
from datetime import datetime

def generate_final_report(video_results, audio_results, text_results, metrics, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "video": video_results,
        "audio": audio_results,
        "text": text_results,
        "overall_score": metrics["overall_score"],
        "approved": metrics["approved"],
        "feedback": metrics["feedback"],
        "timestamp": timestamp
    }
    os.makedirs(output_dir, exist_ok=True)
    json_path = f"{output_dir}/final_report_{timestamp}.json"
    txt_path = f"{output_dir}/final_report_{timestamp}.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Relatório Final - {timestamp}\n")
        f.write(f"Score Geral: {metrics['overall_score']*100:.0f}%\n")
        f.write(f"Status: {'APROVADO' if metrics['approved'] else 'REPROVADO'}\n\n")
        f.write("Feedback:\n")
        for fb in metrics["feedback"]:
            f.write(f"- {fb}\n")
    return {"json_path": json_path, "txt_path": txt_path}