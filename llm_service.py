import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

MODEL_PATH = "/workspace/models/Meta-Llama-3-8B"
llm_pipeline = None

def load_model():
    """
    애플리케이션 시작 시 Llama-3 모델을 로드하고 파이프라인을 초기화합니다.
    4비트 양자화를 사용하여 메모리 사용량을 최적화합니다.
    """
    global llm_pipeline
    if llm_pipeline is not None:
        print("✅ 모델이 이미 로드되었습니다.")
        return

    print("🧠 Llama-3 모델을 로드합니다...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_new_tokens=1024,
        repetition_penalty=1.1,
    )
    print("✅ 모델 로드 완료!")

def get_recommendations(prompt: str) -> str:
    """
    주어진 프롬프트를 사용하여 LLM을 통해 수원지 추천을 생성합니다.
    """
    if llm_pipeline is None:
        raise RuntimeError("모델이 로드되지 않았습니다. 먼저 load_model()을 호출하세요.")

    print(f"💬 추천 생성 프롬프트: {prompt[:100]}...") # 로그에 프롬프트 일부만 출력
    
    # 파이프라인에 토크나이저가 포함되어 있으므로 직접 접근 가능
    eos_token_id = llm_pipeline.tokenizer.eos_token_id

    sequences = llm_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=eos_token_id,
    )
    
    if sequences and len(sequences) > 0:
        # 생성된 텍스트만 반환하도록 수정
        return sequences[0]['generated_text'].replace(prompt, "").strip()
    return "" 