#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
에티오피아 케벨레(Level 4) 경계에 WorldPop 2025 인구 데이터를 합산하여
GeoPackage 및 CSV 파일로 저장하는 스크립트.
"""

import argparse
import logging
import sys
from pathlib import Path
import subprocess
import zipfile
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from tqdm import tqdm
import requests

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)

# --- 상수 정의 ---
# 데이터 출처 URL (원본 링크로 복원)
KEBELE_URL = "https://data.humdata.org/dataset/cod-ab-eth/resource/63c4a9af-53a7-455b-a4d2-adcc22b48d28/download/eth_adm_csa_bofedb_2021_SHP.zip"
WORLDPOP_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2025/2025/ETH/eth_ppp_2025_constrained.tif"

# 기본 디렉터리 및 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
KEBELE_DIR = DATA_DIR / "kebele"
WORLDPOP_DIR = DATA_DIR / "worldpop"
KEBELE_ZIP_PATH = KEBELE_DIR / "Ethiopia_kebele.zip"
WORLDPOP_RASTER_PATH = WORLDPOP_DIR / "eth_ppp_2025_constrained.tif"
OUTPUT_GPKG_PATH = OUTPUT_DIR / "ETH_kebele_pop_2025.gpkg"
OUTPUT_CSV_PATH = OUTPUT_DIR / "ETH_kebele_pop_2025.csv"


def parse_args():
    """CLI 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="에티오피아 케벨레별 인구 데이터를 계산하고 저장합니다."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="로컬에 데이터 파일이 있을 경우 다운로드를 건너뜁니다.",
    )
    parser.add_argument(
        "--reproject",
        type=str,
        default=None,
        metavar="EPSG_CODE",
        help="계산 전 Shapefile을 지정된 EPSG 코드로 재투영합니다. 예: 32637",
    )
    return parser.parse_args()


def download_file(url: str, out_path: Path):
    """지정된 URL에서 파일을 다운로드하고 진행률을 표시합니다."""
    logging.info(f"'{out_path.name}' 다운로드 시작...")
    try:
        # User-Agent를 지정하여 일부 서버의 차단을 우회
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, stream=True, timeout=300, headers=headers)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(out_path, "wb") as f, tqdm(
            desc=out_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        logging.info(f"'{out_path.name}' 다운로드 완료.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"다운로드 실패: {e}")
        # 실패 시 빈 파일 삭제
        if out_path.exists():
            out_path.unlink()
        return False


def get_kebele_gdf(zip_path: Path, reproject_to: str = None) -> gpd.GeoDataFrame:
    """
    압축 해제 없이 ZIP 파일에서 케벨레 Shapefile을 GeoDataFrame으로 로드하고,
    필요 시 재투영합니다.
    """
    logging.info(f"'{zip_path.name}'에서 케벨레 경계 데이터 로드 중...")
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        logging.error(f"케벨레 ZIP 파일이 비어있거나 존재하지 않습니다: '{zip_path}'")
        sys.exit(1)
        
    try:
        # ZIP 내의 .shp 파일을 직접 읽기
        gdf = gpd.read_file(f"zip://{zip_path}")
        
        # 기본 CRS: WorldPop 데이터와 일치시키기 위해 EPSG:4326으로 설정
        target_crs = "EPSG:4326"
        if reproject_to:
            try:
                # 유효한 EPSG 코드인지 확인
                reproject_to_int = int(reproject_to)
                target_crs = f"EPSG:{reproject_to_int}"
            except ValueError:
                logging.error(f"잘못된 EPSG 코드입니다: '{reproject_to}'. 숫자로 입력해주세요.")
                sys.exit(1)

        if gdf.crs != target_crs:
            logging.info(f"데이터를 '{gdf.crs}'에서 '{target_crs}' (으)로 재투영합니다.")
            gdf = gdf.to_crs(target_crs)
            
        logging.info("케벨레 데이터 로드 및 처리 완료.")
        return gdf
        
    except (zipfile.BadZipFile, FileNotFoundError) as e:
        logging.error(f"케벨레 파일을 읽는 데 실패했습니다: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"GeoDataFrame 생성 중 예외 발생: {e}")
        sys.exit(1)


def calc_zonal(gdf: gpd.GeoDataFrame, raster_path: Path) -> gpd.GeoDataFrame:
    """Zonal Statistics를 사용하여 GeoDataFrame의 각 폴리곤 내 인구를 계산합니다."""
    logging.info(f"Zonal statistics 계산 시작 (래스터: '{raster_path.name}')...")
    logging.info("이 작업은 폴리곤의 복잡도에 따라 시간이 다소 소요될 수 있습니다.")
    
    if not raster_path.exists() or raster_path.stat().st_size == 0:
        logging.error(f"WorldPop 래스터 파일이 비어있거나 존재하지 않습니다: '{raster_path}'")
        sys.exit(1)

    try:
        with rasterio.open(raster_path) as src:
            if gdf.crs != src.crs:
                 logging.warning(f"벡터(EPSG:{gdf.crs.to_epsg()})와 래스터(EPSG:{src.crs.to_epsg()})의 CRS가 다릅니다. "
                                 "정확한 계산을 위해 CRS를 일치시키는 것을 권장합니다.")
            
            stats = zonal_stats(
                gdf,
                str(raster_path),
                stats=["sum"],
                nodata=-99999,
                all_touched=True,
                geojson_out=False
            )
            
            gdf["pop_2025_wp"] = [s["sum"] if s["sum"] is not None else 0 for s in stats]
            
            logging.info("Zonal statistics 계산 완료.")
            return gdf
    except Exception as e:
        logging.error(f"Zonal statistics 계산 중 오류 발생: {e}")
        sys.exit(1)
        

def save_outputs(gdf: gpd.GeoDataFrame, out_dir: Path):
    """결과 GeoDataFrame을 GeoPackage와 CSV 파일로 저장합니다."""
    logging.info("결과 파일 저장 시작...")
    
    try:
        gdf.to_file(
            OUTPUT_GPKG_PATH,
            layer="kebele_pop_2025",
            driver="GPKG",
            engine="pyogrio" if "pyogrio" in sys.modules else "fiona"
        )
        logging.info(f"GeoPackage 파일 저장 완료: '{OUTPUT_GPKG_PATH}'")

        gdf.drop(columns="geometry").to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        logging.info(f"CSV 파일 저장 완료: '{OUTPUT_CSV_PATH}'")
        
    except Exception as e:
        logging.error(f"결과 저장 중 오류 발생: {e}")
        sys.exit(1)


def main():
    """메인 실행 함수"""
    args = parse_args()

    # --- 1. 디렉터리 생성 ---
    logging.info("필요한 디렉터리 구조를 확인 및 생성합니다.")
    KEBELE_DIR.mkdir(parents=True, exist_ok=True)
    WORLDPOP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 2. 데이터 다운로드 ---
    if not args.skip_download or not KEBELE_ZIP_PATH.exists():
        download_file(KEBELE_URL, KEBELE_ZIP_PATH)
    else:
        logging.info(f"'{KEBELE_ZIP_PATH.name}' 다운로드 건너뛰기 (파일이 이미 존재함).")

    if not args.skip_download or not WORLDPOP_RASTER_PATH.exists():
        download_file(WORLDPOP_URL, WORLDPOP_RASTER_PATH)
    else:
        logging.info(f"'{WORLDPOP_RASTER_PATH.name}' 다운로드 건너뛰기 (파일이 이미 존재함).")
        
    # --- 3. 데이터 처리 ---
    kebele_gdf = get_kebele_gdf(KEBELE_ZIP_PATH, args.reproject)
    pop_gdf = calc_zonal(kebele_gdf, WORLDPOP_RASTER_PATH)
    
    # --- 4. 결과 저장 ---
    save_outputs(pop_gdf, OUTPUT_DIR)
    
    logging.info(f"\n✅ 완료: {OUTPUT_GPKG_PATH}\n✅ 완료: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main() 