import logging
import os
from pathlib import Path
import requests
import io
from urllib.parse import quote
from sodapy import Socrata
from ingest import ingest_pipeline
import json
import re

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from werkzeug.utils import secure_filename

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)

# Flask 앱 초기화
app = Flask(__name__)
# flash 메시지를 사용하려면 secret_key가 필요합니다.
app.secret_key = "super-secret-key"

# 설정
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 업로드 폴더가 없으면 생성
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# LLM 및 수원지 데이터 로드
water_points_data = []
try:
    # 수원지 데이터 로드
    with open('static/ethiopia_water_points.json', 'r') as f:
        water_points_data = json.load(f)
    logging.info(f"✅ 수원지 데이터 {len(water_points_data)}개 로드 완료")
except Exception as e:
    logging.error(f"초기화 중 오류 발생: {e}", exc_info=True)


def allowed_file(filename: str) -> bool:
    """파일 확장자가 허용된 목록에 있는지 확인합니다."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    """메인 UI 페이지를 렌더링합니다."""
    return render_template("index.html")


@app.route("/analysis", methods=["GET"])
def analysis():
    """분석 페이지를 렌더링합니다."""
    return render_template("analysis.html")


@app.route("/api/waterpoints", methods=["GET"])
def get_water_points():
    """미리 처리된 에티오피아 수자원 데이터를 JSON 파일에서 직접 제공합니다."""
    return jsonify(water_points_data)


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    파일을 업로드하고 인덱싱 파이프라인을 실행합니다.
    UI와 API 요청 모두를 처리할 수 있도록 수정합니다.
    """
    if "file" not in request.files:
        flash("요청에 파일 부분이 없습니다.", "error")
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        flash("선택된 파일이 없습니다.", "error")
        return redirect(request.url)

    if file and file.filename and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = Path(app.config["UPLOAD_FOLDER"]) / filename
        
        try:
            logging.info(f"Saving uploaded file to {file_path}")
            file.save(file_path)

            logging.info(f"Starting ingestion pipeline for {filename}")
            success = ingest_pipeline([file_path])

            if success:
                flash(f"파일 '{filename}'이(가) 성공적으로 처리되었습니다.", "success")
            else:
                flash(f"파일 '{filename}' 처리 중 오류가 발생했습니다.", "error")

        except Exception as e:
            logging.error(f"업로드 및 처리 중 예외 발생: {e}", exc_info=True)
            flash(f"예기치 않은 오류가 발생했습니다: {str(e)}", "error")
        finally:
            if file_path.exists():
                try:
                    os.remove(file_path)
                    logging.info(f"임시 파일 삭제: {file_path}")
                except OSError as e:
                    logging.error(f"파일 삭제 오류 {file_path}: {e}")
        
        return redirect(url_for("index"))

    else:
        flash(f"허용되지 않는 파일 형식입니다. {list(ALLOWED_EXTENSIONS)} 형식만 가능합니다.", "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    # 프로덕션 환경에서는 gunicorn과 같은 WSGI 서버를 사용해야 합니다.
    # 예: gunicorn --workers 2 --bind 0.0.0.0:8888 app:app
    app.run(host="0.0.0.0", port=8877, debug=True) 