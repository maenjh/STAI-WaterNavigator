import logging
import os
from pathlib import Path
import requests
import io
from urllib.parse import quote
from sodapy import Socrata
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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)

# Flask 앱 초기화
app = Flask(__name__)
# flash 메시지를 사용하려면 secret_key가 필요합니다.
app.secret_key = "super-secret-key"

# LLM 및 수원지 데이터 로드
water_points_data = []
try:
    # 수원지 데이터 로드
    with open('static/ethiopia_water_points.json', 'r') as f:
        water_points_data = json.load(f)
    logging.info(f"✅ 수원지 데이터 {len(water_points_data)}개 로드 완료")
except Exception as e:
    logging.error(f"초기화 중 오류 발생: {e}", exc_info=True)


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


if __name__ == "__main__":
    # 프로덕션 환경에서는 gunicorn과 같은 WSGI 서버를 사용해야 합니다.
    # 예: gunicorn --workers 2 --bind 0.0.0.0:8888 app:app
    app.run(host="0.0.0.0", port=8877, debug=True) 