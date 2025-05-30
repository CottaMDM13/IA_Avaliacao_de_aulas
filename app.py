from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
from flask_bootstrap import Bootstrap5
import os
import time
import sqlite3
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.video_loader import load_and_extract_features
from src.utils.transcriber import transcribe_audio
from src.analysis.video import analyze_video, evaluate_video_quality
from src.analysis.audio import extract_audio_features, evaluate_audio_quality
from src.analysis.text import analyze_transcriptions
from src.analysis.metrics import calculate_overall_score
from src.utils.report import generate_final_report, generate_docx_report
import yaml
import mimetypes
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
# Configurar tamanho máximo de upload (100MB)
app.config['MAX_CONTENT_LENGTH'] = 3000 * 1024 * 1024
Bootstrap5(app)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Armazenar progresso
progress = {}

def load_config():
    try:
        with open("config/settings.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        logger.error(f"Erro ao carregar config/settings.yaml: {str(e)}")
        raise

@app.route("/", methods=["GET", "POST"])
def index():
    config = load_config()
    if request.method == "POST":
        logger.info("Recebida requisição POST para upload de vídeos")
        
        try:
            if 'videos' not in request.files or not request.files.getlist("videos"):
                logger.error("Nenhum arquivo enviado")
                return jsonify({"error": "Nenhum arquivo enviado."}), 400
            
            video_files = request.files.getlist("videos")
            total_files = len(video_files)
            session_id = str(time.time())
            progress[session_id] = {"completed": 0, "total": total_files}
            report_ids = []
            
            for i, video_file in enumerate(video_files):
                logger.debug(f"Processando: {video_file.filename}")
                
                if not video_file.filename:
                    logger.error("Arquivo sem nome detectado")
                    return jsonify({"error": "Um ou mais arquivos estão sem nome."}), 400
                
                if not video_file.filename.lower().endswith('.mp4'):
                    logger.error(f"Arquivo inválido: {video_file.filename}. Apenas .mp4 é aceito")
                    return jsonify({"error": f"Arquivo {video_file.filename} inválido. Apenas .mp4 é aceito."}), 400
                
                mime_type, _ = mimetypes.guess_type(video_file.filename)
                if mime_type != 'video/mp4':
                    logger.error(f"Tipo MIME inválido para {video_file.filename}: {mime_type}")
                    return jsonify({"error": f"Arquivo {video_file.filename} não é um vídeo MP4 válido."}), 400

                video_path = f"{config['paths']['input_videos']}/{video_file.filename}"
                audio_path = f"{config['paths']['input_audio']}/{video_file.filename.rsplit('.', 1)[0]}.wav"
                output_dir = config["paths"]["output_reports"]

                try:
                    os.makedirs(config["paths"]["input_videos"], exist_ok=True)
                    os.makedirs(config["paths"]["input_audio"], exist_ok=True)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    logger.debug(f"Salvando vídeo em: {video_path}")
                    video_file.save(video_path)
                    
                    start_time = time.time()
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        future_video_features = executor.submit(load_and_extract_features, video_path, audio_path)
                        future_transcription = executor.submit(transcribe_audio, audio_path, config["paths"]["output_transcripts"])
                        future_video_metrics = executor.submit(analyze_video, video_path)
                        
                        video_features = future_video_features.result()
                        transcription = future_transcription.result()
                        video_metrics = future_video_metrics.result()
                    
                    video_results = evaluate_video_quality(video_metrics, config)
                    audio_features = extract_audio_features(audio_path)
                    audio_results = evaluate_audio_quality(audio_features, config)
                    text_results = analyze_transcriptions([transcription["text"]], [video_results], [audio_results], config)[0]
                    metrics = calculate_overall_score(video_results, audio_results, text_results, config)
                    
                    report = generate_final_report(video_results, audio_results, text_results, metrics, output_dir, video_file.filename)
                    processing_time = time.time() - start_time
                    
                    conn = sqlite3.connect(f"{output_dir}/reports.db")
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM reports WHERE video_name = ? AND timestamp = ?", 
                                 (video_file.filename, report['metrics']['timestamp']))
                    report_id = cursor.fetchone()
                    if report_id:
                        report_ids.append(report_id[0])
                    else:
                        logger.error(f"Relatório não encontrado para {video_file.filename}")
                        raise ValueError("Falha ao recuperar ID do relatório")
                    conn.close()
                    
                    progress[session_id]["completed"] = i + 1
                    logger.debug(f"Progresso atualizado: {i+1}/{total_files}")
                    
                except Exception as e:
                    logger.error(f"Erro ao processar {video_file.filename}: {str(e)}")
                    return jsonify({"error": f"Erro ao processar {video_file.filename}: {str(e)}"}), 500
            
            logger.info(f"Processamento concluído para {total_files} vídeos")
            return jsonify({"redirect": url_for('reports'), "session_id": session_id})
        
        except RequestEntityTooLarge:
            logger.error("Arquivo enviado excede o tamanho máximo permitido (100MB)")
            return jsonify({"error": "O vídeo é muito grande. O tamanho máximo permitido é 100MB."}), 413
        except Exception as e:
            logger.error(f"Erro inesperado ao processar upload: {str(e)}")
            return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500
    
    return render_template("index.html")

@app.route("/progress/<session_id>")
def get_progress(session_id):
    try:
        if session_id in progress:
            completed = progress[session_id]["completed"]
            total = progress[session_id]["total"]
            percentage = (completed / total) * 100 if total > 0 else 0
            logger.debug(f"Progresso para session_id {session_id}: {percentage}%")
            return jsonify({"percentage": percentage, "completed": completed, "total": total})
        logger.warning(f"Session_id {session_id} não encontrado")
        return jsonify({"percentage": 0, "completed": 0, "total": 0})
    except Exception as e:
        logger.error(f"Erro na rota /progress: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/reports")
def reports():
    try:
        config = load_config()
        output_dir = config["paths"]["output_reports"]
        conn = sqlite3.connect(f"{output_dir}/reports.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, video_name, score, approved, timestamp FROM reports ORDER BY timestamp DESC")
        raw_reports = cursor.fetchall()
        conn.close()

        reports = []
        for report in raw_reports:
            try:
                timestamp = datetime.strptime(report[4], "%Y%m%d_%H%M%S")
                formatted_timestamp = timestamp.strftime("%d/%m/%Y %H:%M:%S")
            except ValueError:
                formatted_timestamp = report[4]
            reports.append({
                "id": report[0],
                "video_name": report[1],
                "score": report[2],
                "approved": report[3],
                "timestamp": formatted_timestamp
            })

        logger.debug("Relatórios obtidos com sucesso")
        return render_template("reports.html", reports=reports)
    except Exception as e:
        logger.error(f"Erro ao carregar relatórios: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/report/<int:report_id>")
def report(report_id):
    try:
        config = load_config()
        output_dir = config["paths"]["output_reports"]
        conn = sqlite3.connect(f"{output_dir}/reports.db")
        cursor = conn.cursor()
        cursor.execute("SELECT video_name, score, feedback, approved, timestamp FROM reports WHERE id = ?", (report_id,))
        report = cursor.fetchone()
        conn.close()
        
        if not report:
            logger.error(f"Relatório {report_id} não encontrado")
            return render_template("error.html", error="Relatório não encontrado."), 404
        
        import json
        feedback = json.loads(report[2])
        try:
            timestamp = datetime.strptime(report[4], "%Y%m%d_%H%M%S")
            formatted_timestamp = timestamp.strftime("%d/%m/%Y %H:%M:%S")
        except ValueError:
            formatted_timestamp = report[4]
        metrics = {
            "overall_score": report[1],
            "approved": report[3] if report[3] is not None else False,
            "feedback": feedback,
            "timestamp": formatted_timestamp
        }
        json_path = f"{output_dir}/final_report_{report[0]}_{report[4]}.json"
        logger.debug(f"Relatório {report_id} carregado com sucesso")
        return render_template("result.html", metrics=metrics, json_path=json_path, video_name=report[0], report_id=report_id)
    except Exception as e:
        logger.error(f"Erro ao carregar relatório {report_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/download/<int:report_id>")
def download(report_id):
    try:
        config = load_config()
        output_dir = config["paths"]["output_reports"]
        conn = sqlite3.connect(f"{output_dir}/reports.db")
        cursor = conn.cursor()
        cursor.execute("SELECT video_name, score, feedback, approved, timestamp FROM reports WHERE id = ?", (report_id,))
        report = cursor.fetchone()
        conn.close()
        
        if not report:
            logger.error(f"Relatório {report_id} não encontrado para download")
            return render_template("error.html", error="Relatório não encontrado."), 404
        
        import json
        feedback = json.loads(report[2])
        try:
            timestamp = datetime.strptime(report[4], "%Y%m%d_%H%M%S")
            formatted_timestamp = timestamp.strftime("%d/%m/%Y %H:%M:%S")
        except ValueError:
            formatted_timestamp = report[4]
        metrics = {
            "overall_score": report[1],
            "approved": report[3] if report[3] is not None else False,
            "feedback": feedback,
            "timestamp": formatted_timestamp
        }
        docx_path = generate_docx_report(report[0], metrics, output_dir)
        logger.debug(f"Arquivo .docx gerado: {docx_path}")
        return send_file(docx_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Erro ao gerar .docx para relatório {report_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)