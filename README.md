
````markdown
# 🌍 STAI 2025 - WaterNavigator

WaterNavigator는 STAI 2025에서 개발한 Flask 기반 데이터 분석 웹 애플리케이션입니다.  
에티오피아·이집트 등 최빈개발국(LDC)에서 물 부족 문제와 물 분쟁을 완화하기 위해,  
데이터 기반으로 최적의 우물 설치 위치를 추천하고, NGO 및 정부의 의사결정을 지원합니다.

---

## 🚀 프로젝트 실행 방법

### 1️⃣ 환경 준비

```bash
# 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
````

### 2️⃣ Flask 앱 실행

```bash
python app.py
```

실행 후, 브라우저에서 아래 주소로 접속:

```
http://localhost:5000/
```

---

## 🏗️ 프로젝트 구조

```
.
├── app.py                  # Flask 메인 서버
├── requirements.txt        # Python 패키지 목록
├── /static                # CSS, JS, 이미지 파일
├── /templates             # HTML 템플릿 (예: Jinja2)
├── /data                 # 원본 데이터 (.tiff, .csv)
├── /scripts              # 데이터 전처리, 분석 코드
├── /ui                  # 웹 UI 코드 (예: React 연동 시)
└── /infra               # AWS 배포 스크립트
```

---

## 💡 핵심 기능

* **데이터 기반 우물 설치 위치 추천**

  * 인구 밀도, 가뭄 지수, 지하수 위치, 농경지 면적, 기존 급수 시설 분석
* **최적 위치 추천 보고서 생성**
* **분쟁/민감 지역 사전 경고**
* **Web UI 지도 시각화 및 추천 근거 제공**

---

## 🌐 주요 기술 스택

* Python 3.x
* Flask
* Pandas, Numpy
* AWS EC2, S3
* (선택) React, Leaflet.js (지도 시각화)

---

## 📊 데이터 설명

| 데이터 항목     | 설명                       |
| ---------- | ------------------------ |
| 인구 데이터     | 행정 구역별 인구 밀도, 분포         |
| 기존 수자원 인프라 | 우물, 펌프, 급수 시설 위치 정보      |
| 지하수 수위     | 지질 정보 기반 지하수 잠재성 점수      |
| 농경지 면적     | 토지이용 지도상 경작지 및 정착지 범위    |
| 가뭄 수치      | 복합 가뭄 지수(CDI), 취약 지역 가중치 |

---

## 🎯 기대 효과

* 주민 물 수집 시간 단축
* 가뭄 회복력 및 식량 안보 개선
* 소규모 농업, 가축 사육 물 접근성 개선
* NGO 및 정부의 데이터 기반 의사결정 최적화

---

## 🏛️ 주요 고객 및 협력 대상

* United Nations Development Programme (UNDP)
* Green Climate Fund (GCF)
* World Vision
* UNICEF WASH

---

## 🙌 기여 방법

1. 이 레포를 fork 후 작업
2. pull request로 개선 사항 제출
3. issue에 버그 또는 개선 요청 등록



---

