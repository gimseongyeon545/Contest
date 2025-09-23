# inference.py
# - streamlit_app.py가 import 해서 infer(question: str) -> str 호출
# - 단독 실행하면 test.csv -> submission.csv 배치 추론 수행
#   (사용법) python inference.py

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# ==== train 모듈의 함수 불러오기 ====
from train import (
    warmup_from_manifest,
    warmup_retrievers,              # 선택적(없으면 try/except 아래에서 무시)
    build_context_for_question,
    add_ctx_to_prompt,
    extract_answer_only,
    is_multiple_choice,
    extract_question_and_choices,
)


# =========================
# 전역 캐시 (첫 호출 1회 로드)
# =========================
_PIPE = None
_WARMED = False
_MANIFEST_PATH = (Path(".").resolve() / "artifacts" / "manifest.json")

# =========================
# 모델 로더
# =========================
def _build_pipe(model_name: str = "K-intelligence/Midm-2.0-Base-Instruct"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = {}
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs.update(dict(
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        ))
    except Exception:
        model_kwargs.update(dict(
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        ))

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return gen_pipe

# =========================
# 프롬프트
# =========================
def _make_prompt_auto(text: str) -> str:
    if is_multiple_choice(text):
        question, _nums, _contents, options = extract_question_and_choices(text)
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 [실제 질문]에 대해 가장 적절한 **정답 선택지 번호 한 개만 출력**하세요.\n"
            "절대 미응답 하지 마세요.\n"
            f"[실제 질문]: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n"
            "절대 미응답 하지 마세요.\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )
    return prompt

# =========================
# 1회 초기화
# =========================
def _ensure_inited(manifest_path: Optional[str | Path] = None):
    """
    - manifest.json 로드 → 각 인덱스 경로 세팅 → retriever warmup
    - 생성 모델 파이프라인 로드
    """
    global _PIPE, _WARMED, _MANIFEST_PATH
    if _PIPE is not None and _WARMED:
        return

    mp = Path(manifest_path) if manifest_path else _MANIFEST_PATH
    if not mp.exists():
        raise FileNotFoundError(
            f"manifest not found: {mp}\n"
            "-> 먼저 train.py를 실행해서 artifacts/manifest.json을 생성하세요."
        )

    # 인덱스/임베딩 warmup (모든 코퍼스 켜둠)
    warmup_from_manifest(
        str(mp),
        cache_decrees=True,
        cache_rules=True,
        cache_regs=True,
        cache_oper=True,
    )
    # (보수적) 추가 예열이 필요한 경우 시도 — 없으면 무시
    try:
        warmup_retrievers(cache_decrees=True, cache_rules=True, cache_regs=True, cache_oper=True)
    except Exception:
        pass

    _PIPE = _build_pipe()
    _WARMED = True

# =========================
# 공개 API: infer(question) -> str
# =========================
def infer(question: str) -> str:
    """
    하나의 질문을 받아 하나의 답변(문자열) 반환.
    streamlit_app.py에서 infer(user_input)을 호출.
    """
    _ensure_inited()

    # 컨텍스트 구성(RAG)
    ctx = build_context_for_question(question, char_budget=4000)

    # 프롬프트 결합
    final_prompt = add_ctx_to_prompt(_make_prompt_auto(question), ctx)

    # 생성
    out = _PIPE(
        final_prompt,
        max_new_tokens=512,
        temperature=0.0,  # 결정론적 추론
        top_p=1.0,
        do_sample=False,
    )
    text_out = out[0]["generated_text"]

    # 후처리: 숫자만/빈값 방지
    answer = extract_answer_only(text_out, original_question=question).strip()
    return answer if answer else "미응답"

# =========================
# 배치 실행 (CLI용)
# =========================
if __name__ == "__main__":
    # 기본 경로(현재 디렉토리)
    BASE = Path(".").resolve()
    manifest_path = BASE / "artifacts" / "manifest.json"
    test_csv     = BASE / "test.csv"
    sample_csv   = BASE / "sample_submission.csv"
    out_csv      = BASE / "submission.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found: {test_csv}")
    if not sample_csv.exists():
        raise FileNotFoundError(f"sample_submission.csv not found: {sample_csv}")

    # 초기화
    _ensure_inited(manifest_path)

    # 배치 추론
    test_df = pd.read_csv(test_csv)
    preds = []
    for q in tqdm(test_df["Question"], desc="Inference"):
        preds.append(infer(str(q)))

    # 저장
    sample_df = pd.read_csv(sample_csv)
    sample_df["Answer"] = preds[: len(sample_df)]
    sample_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved submission -> {out_csv}")