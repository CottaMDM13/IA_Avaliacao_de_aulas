import json
import os
from datetime import datetime

def generate_final_report(video_results, audio_results, text_results, metrics, output_dir, video_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "video": video_results,
        "audio": audio_results,
        "text": text_results,
        "overall_score": metrics["overall_score"],
        "approved": metrics["approved"],
        "feedback": metrics["feedback"],
        "timestamp": timestamp,
        "video_name": video_name
    }
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"final_report_{video_name.rsplit('.', 1)[0]}_{timestamp}"
    json_path = f"{output_dir}/{base_name}.json"
    txt_path = f"{output_dir}/{base_name}.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Relatório Final - {video_name} - {timestamp}\n")
        f.write(f"Score Geral: {metrics['overall_score']*100:.0f}%\n")
        f.write(f"Status: {'APROVADO' if metrics['approved'] else 'REPROVADO'}\n\n")
        f.write("Resumo:\n")
        f.write(f"{text_results.get('review', 'Sem resumo disponível.')}\n\n")
        f.write("Feedback Técnico:\n")
        for fb in metrics["feedback"]:
            f.write(f"- {fb}\n")
    return {"json_path": json_path, "txt_path": txt_path}