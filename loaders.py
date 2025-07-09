import logging
from pathlib import Path
from typing import List, Dict, Callable, Any

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 확장자별 로더 매핑
# .doc와 .docx 모두 UnstructuredWordDocumentLoader를 사용합니다.
LOADER_MAPPING: Dict[str, Callable[[str], Any]] = {
    ".pdf": UnstructuredPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
}


def load_documents(file_path: Path) -> List[Document]:
    """
    파일 경로와 확장자에 따라 적절한 로더를 사용하여 문서를 로드합니다.

    Args:
        file_path (Path): 처리할 파일의 경로.

    Returns:
        List[Document]: 로드된 문서의 리스트.

    Raises:
        ValueError: 지원되지 않는 파일 확장자인 경우 발생합니다.
        Exception: 문서 로딩 중 발생하는 모든 예외를 다시 발생시킵니다.
    """
    ext = file_path.suffix.lower()
    if ext not in LOADER_MAPPING:
        logging.error(f"Unsupported file extension: '{ext}'")
        raise ValueError(f"Unsupported file extension: {ext}")

    loader_class = LOADER_MAPPING[ext]
    loader = loader_class(str(file_path))

    try:
        logging.info(f"Loading document: {file_path}")
        # load() 메서드는 문서 페이지별로 Document 객체를 리스트에 담아 반환합니다.
        return loader.load()
    except Exception as e:
        logging.error(f"Error loading document {file_path}: {e}", exc_info=True)
        # 예외를 상위 호출자로 전파하여 처리하도록 합니다.
        raise


def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    로드된 문서를 지정된 크기의 청크로 분할합니다.

    Args:
        documents (List[Document]): 분할할 Document 객체의 리스트.
        chunk_size (int): 각 청크의 최대 문자 수.
        chunk_overlap (int): 인접한 청크 간의 겹치는 문자 수.

    Returns:
        List[Document]: 분할된 청크(Document 객체)의 리스트.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    logging.info(f"Splitting {len(documents)} document(s) into chunks.")
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Successfully created {len(chunks)} chunks.")
    return chunks 