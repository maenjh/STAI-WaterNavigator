import argparse
import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from loaders import load_documents, split_documents

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)

# 상수 정의
FAISS_INDEX_PATH = "faiss_rag_index"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """
    지정된 이름의 HuggingFace 임베딩 모델을 로드합니다.
    sentence-transformers는 CUDA가 사용 가능하면 자동으로 GPU를 사용합니다.
    """
    logging.info(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


def process_and_chunk_files(file_paths: List[Path]) -> List[Document]:
    """
    주어진 파일 경로 리스트를 순회하며 문서를 로드하고 청크로 분할합니다.
    """
    all_chunks = []
    for file_path in tqdm(file_paths, desc="Processing files"):
        if not file_path.is_file():
            logging.warning(f"File not found: {file_path}. Skipping.")
            continue
        try:
            documents = load_documents(file_path)
            chunks = split_documents(documents)
            # 각 청크에 원본 파일 이름 메타데이터 추가
            for chunk in chunks:
                chunk.metadata["source"] = str(file_path.name)
            all_chunks.extend(chunks)
            logging.info(f"Successfully processed and chunked {file_path.name}")
        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}", exc_info=True)
    return all_chunks


def ingest_pipeline(file_paths: List[Path]) -> bool:
    """
    문서 인덱싱 전체 파이프라인을 실행하고 성공 여부를 반환합니다.
    app.py에서 호출하기 위한 함수입니다.
    """
    index_path = Path(FAISS_INDEX_PATH)

    embeddings = get_embeddings(EMBEDDING_MODEL_NAME)
    new_chunks = process_and_chunk_files(file_paths)

    if not new_chunks:
        logging.warning("No new documents were processed successfully.")
        return False

    main_index: Optional[FAISS] = None
    if index_path.exists() and any(index_path.iterdir()):
        try:
            logging.info(f"Loading existing FAISS index from '{index_path}'")
            # LangChain의 최신 보안 정책에 따라 `allow_dangerous_deserialization` 필요
            main_index = FAISS.load_local(
                str(index_path), embeddings, allow_dangerous_deserialization=True
            )
            logging.info("Successfully loaded existing index.")
        except Exception as e:
            logging.error(f"Could not load existing index: {e}. A new index will be created.", exc_info=True)
            # 로드 실패 시, 아래에서 새로운 인덱스를 생성합니다.
    
    # 4. 인덱스에 새로운 데이터 추가 또는 신규 생성
    if main_index:
        # 기존 인덱스가 있을 경우: 새로운 청크로 임시 인덱스를 만들어 병합
        logging.info(f"Creating a temporary index for {len(new_chunks)} new chunks.")
        temp_index = FAISS.from_documents(new_chunks, embeddings)
        logging.info("Merging new documents into the existing index.")
        main_index.merge_from(temp_index)
    else:
        # 기존 인덱스가 없을 경우: 새로운 청크로 인덱스를 처음부터 생성
        logging.info("Creating a new FAISS index from scratch.")
        main_index = FAISS.from_documents(new_chunks, embeddings)

    # 5. 업데이트된 인덱스 저장
    logging.info(f"Saving updated index to '{index_path}'")
    main_index.save_local(str(index_path))
    logging.info("Ingestion process completed successfully.")
    return True

def main_cli():
    """
    CLI 실행을 위한 메인 함수입니다.
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents (.pdf, .docx, .doc) into a FAISS vector store."
    )
    parser.add_argument(
        "files", nargs="+", help="List of file paths to ingest."
    )
    args = parser.parse_args()
    file_paths = [Path(p) for p in args.files]
    ingest_pipeline(file_paths)


if __name__ == "__main__":
    main_cli() 