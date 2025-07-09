#!/usr/bin/env python3
"""
Llama-3, FAISS, Tavily API를 사용한 RAG 시스템
"""

import os
import torch
import warnings
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain import hub

# 경고 메시지 무시
warnings.filterwarnings("ignore")

def create_rag_system():
    """RAG 시스템을 생성하고 실행합니다."""
    
    # 1. 모델 및 토크나이저 설정
    model_path = "/workspace/models/Meta-Llama-3-8B"
    embedding_model_name = "jhgan/ko-sroberta-multitask"
    
    print("🤖 RAG 시스템을 초기화합니다...")

    # 4-bit 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"🧠 Llama-3 모델을 로드합니다... (경로: {model_path})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # LangChain 파이프라인 생성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_new_tokens=512,
        repetition_penalty=1.1,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ 모델 로드 완료!")

    # 2. 임베딩 모델 설정
    print(f"🖼️  임베딩 모델을 로드합니다... (모델: {embedding_model_name})")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print("✅ 임베딩 모델 로드 완료!")

    # 3. 로컬 벡터 DB (FAISS) 생성
    print("📚 로컬 벡터 DB를 생성합니다...")
    # 예제 문서
    docs = [
        "인공지능(AI)은 컴퓨터가 인간처럼 학습하고, 추론하고, 문제를 해결할 수 있도록 하는 기술입니다.",
        "RAG (Retrieval-Augmented Generation)는 대규모 언어 모델(LLM)이 외부 지식 베이스의 정보를 검색하여 답변의 정확성과 신뢰성을 높이는 기술입니다.",
        "FAISS는 Facebook AI Research에서 개발한 효율적인 유사도 검색 및 밀집 벡터 클러스터링 라이브러리입니다.",
        "대한민국의 수도는 서울입니다. 서울은 경제, 문화, 교통의 중심지입니다."
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.create_documents(docs)
    
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    print("✅ 벡터 DB 생성 완료!")
    
    # 4. 도구(Tools) 정의
    print("🛠️  도구를 정의합니다...")
    # 4-1. 로컬 DB 검색 도구
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retrieval_chain = retriever | format_docs
    
    class LocalSearchInput(BaseModel):
        query: str = Field(description="로컬 지식 베이스에서 검색할 쿼리")

    class LocalSearchTool(BaseTool):
        name: str = "local_search"
        description: str = "AI, RAG, FAISS, 한국 수도 등 내부 정보에 대해 질문할 때 사용하세요."
        args_schema: type[BaseModel] = LocalSearchInput

        def _run(self, query: str) -> str:
            """Use the tool."""
            docs = retrieval_chain.invoke(query)
            return docs if docs else "검색 결과가 없습니다."

    # 4-2. 웹 검색 도구 (Tavily)
    tavily_search = TavilySearchResults(max_results=3)
    print("✅ 도구 정의 완료!")
    
    tools = [
        LocalSearchTool(),
        Tool(
            name="tavily_search",
            description="실시간 정보, 최신 뉴스, 특정 인물/장소/사건 등 웹 검색이 필요한 질문에 사용하세요.",
            func=tavily_search.invoke
        )
    ]
    
    # 5. 에이전트 생성
    print("🧑‍✈️ 에이전트를 생성합니다...")
    # ReAct 프롬프트 가져오기
    prompt = hub.pull("hwchase17/react")
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    print("✅ 에이전트 생성 완료!")
    return agent_executor

def main():
    """메인 실행 함수"""
    if "TAVILY_API_KEY" not in os.environ:
        print("❌ TAVILY_API_KEY 환경 변수가 설정되지 않았습니다. API 키를 설정해주세요.")
        return

    agent_executor = create_rag_system()

    # 명령줄 인자가 있으면 해당 인자를 질문으로 사용
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print("\n" + "="*50)
        print(f"💬 질문: {query}")
        print("="*50)
        try:
            response = agent_executor.invoke({"input": query})
            print("\n💡 답변:", response['output'])
        except Exception as e:
            print(f"❌ 오류가 발생했습니다: {e}")
        return

    # 대화형 모드
    print("\n" + "="*50)
    print("🤖 RAG 시스템이 준비되었습니다. 질문을 입력하세요. (종료: 'exit')")
    print("="*50)
    
    while True:
        query = input("💬 질문: ")
        if query.lower() == 'exit':
            break
        
        try:
            response = agent_executor.invoke({"input": query})
            print("\n💡 답변:", response['output'])
        except Exception as e:
            print(f"❌ 오류가 발생했습니다: {e}")
        print("\n" + "-"*50)

if __name__ == "__main__":
    main() 