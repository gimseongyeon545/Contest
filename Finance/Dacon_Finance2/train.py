# -*- coding: utf-8 -*-
"""
train.py  (import-safe, full refactor)

- 전역에는 "정의"만 존재: 함수/클래스/상수/정규식 등
- 무거운 작업(파일 생성/인덱싱/저장/프린트)은 모두 main() 내부에서만 호출
- inference.py에서 `from train import ...`로 임포트해도 어떤 빌드/저장은 실행되지 않음

실행:
    python train.py                # (기본: 현재 디렉토리를 BASE_ROOT로 사용)
    python train.py --base_root ./ # 동일
"""

# ==============================
# Imports (모듈 정의 전부 상단)
# ==============================
import os
import re
import glob
import json
import math
import gc
import hashlib
import textwrap
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from itertools import chain, islice
from functools import lru_cache
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================
# 공통 유틸 / 정규식
# ==============================
import random  # ← 추가


def set_seed(seed: int = 208):
    """학습/인덱싱 재현성 확보용 시드 고정"""
    os.environ["PYTHONHASHSEED"] = str(seed)          
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)

# ==============================
# 공통 유틸 / 정규식
# ==============================
def _ensure_dirs(*paths: Path):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def _label_tag(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace(".", "-").replace(",", "_")
    return s or "무번호"

# --- 3단비교/FAMILY 관련 정규식 ---
ART_PAT = re.compile(r"제\d+조(?:의\d+)?")
HEADER_PAT = re.compile(r"(제\d+조(?:의\d+)?\([^)]*\))")

def extract_first_article(text: str):
    if not isinstance(text, str): return np.nan
    m = ART_PAT.search(text)
    return m.group(0) if m else np.nan

def normalize_spaces(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def article_key_from_header(header: str) -> str:
    if not isinstance(header, str): return None
    m = ART_PAT.search(header)
    return m.group(0) if m else None

def extract_article_blocks(text: str):
    """
    본문 문자열에서 조문 헤더 블록(가장 긴 버전 우선)을 추출
    """
    blocks = {}
    if not isinstance(text, str): return blocks
    text = normalize_spaces(text)
    headers = list(HEADER_PAT.finditer(text))
    if headers:
        for i, m in enumerate(headers):
            start = m.start()
            end = headers[i+1].start() if i+1 < len(headers) else len(text)
            block = text[start:end].strip()
            key = article_key_from_header(m.group(0))
            if key:
                if key not in blocks or len(blocks[key]) < len(block):
                    blocks[key] = block
    else:
        arts = list(dict.fromkeys(ART_PAT.findall(text)))
        for a in arts:
            if a not in blocks or len(blocks[a]) < len(text):
                blocks[a] = text
    return blocks

def merge_blocks_maxlen(dicts):
    out = {}
    for d in dicts:
        if not isinstance(d, dict): continue
        for k, v in d.items():
            if not isinstance(v, str): continue
            if k not in out or len(out[k]) < len(v):
                out[k] = v
    return out

def key_num(s: str) -> int:
    m = re.search(r"\d+", s or "")
    return int(m.group(0)) if m else 10**9

# ==============================
# (A) 3단비교 HTML → CSV 생성
# ==============================
def html_to_csv(path: str, out_csv_path: str):
    """
    - path: HTML/XLS 파일 경로 (read_html로 읽힘)
    - out_csv_path: 저장할 CSV 경로 (예: ./3단비교/개인정보보호법/개인인용.csv)
    """
    tables = pd.read_html(path, flavor="lxml")
    df = max(tables, key=lambda d: d.shape[0])
    # 보통 숫자 인덱스 컬럼들(0,1,2)에 유의미 데이터가 있음
    num_cols = [c for c in df.columns if isinstance(c, (int, float)) or str(c).isdigit()]
    df = df[num_cols[:3]]
    df.columns = ["법령", "시행령", "시행규칙"][:df.shape[1]]
    df = df.dropna(how="all").reset_index(drop=True)
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    print(f"[OK] 3단 CSV saved: {out_csv_path}")

def build_3dan_csvs(base_root: Path):
    """
    ./3단비교/{법령}/(인용조문 3단비교).xls, (위임조문 3단비교).xls → CSV로 저장
    """
    dict_ = {
        '개인정보보호법':'개인', '신용정보법':'신용', '자본시장법':'자본',
        '전자금융거래법':'금융', '전자서명법':'서명', '정보통신망법':'통신'
    }
    tri_dir = base_root / "3단비교"
    for key, val in dict_.items():
        in_dir = tri_dir / key
        in_html_inyong = in_dir / "(인용조문 3단비교).xls"
        in_html_wiim   = in_dir / "(위임조문 3단비교).xls"
        out_csv_inyong = tri_dir / key / f"{val}인용.csv"
        out_csv_wiim   = tri_dir / key / f"{val}위임.csv"
        if in_html_inyong.exists():
            html_to_csv(str(in_html_inyong), str(out_csv_inyong))
        else:
            print(f"[WARN] not found: {in_html_inyong}")
        if in_html_wiim.exists():
            html_to_csv(str(in_html_wiim), str(out_csv_wiim))
        else:
            print(f"[WARN] not found: {in_html_wiim}")

# ==============================
# (B) FAMILY CSV 빌드 (연결_컨텐츠 포함, 'same-num'/'cross-num' 구성)
# ==============================
def build_family_csv(root_dir: Path, shortname: str):
    """
    ./3단비교/{법령}/ {shortname}인용.csv, {shortname}위임.csv → {shortname}_family.csv 생성
    - 연결_컨텐츠: 시행령/시행규칙 조문 텍스트를 JSON list로 저장
    """
    path_wiim   = root_dir / f"{shortname}위임.csv"
    path_inyong = root_dir / f"{shortname}인용.csv"
    out_path    = root_dir / f"{shortname}_family.csv"

    if not path_wiim.exists() or not path_inyong.exists():
        print(f"[WARN] family CSV source not found (skip): {root_dir}")
        return

    df_wiim   = pd.read_csv(path_wiim)
    df_inyong = pd.read_csv(path_inyong)
    for df in (df_wiim, df_inyong):
        for col in ["법령","시행령","시행규칙"]:
            if col not in df.columns:
                df[col] = np.nan

    # 파생 컬럼
    for df in (df_wiim, df_inyong):
        df["법령_key"]          = df["법령"].apply(extract_first_article)
        df["시행령_arts"]       = df["시행령"].apply(lambda s: list(dict.fromkeys(ART_PAT.findall(s))) if isinstance(s,str) else [])
        df["시행규칙_arts"]     = df["시행규칙"].apply(lambda s: list(dict.fromkeys(ART_PAT.findall(s))) if isinstance(s,str) else [])
        df["시행령_blocks"]     = df["시행령"].apply(extract_article_blocks)
        df["시행규칙_blocks"]   = df["시행규칙"].apply(extract_article_blocks)

    df_all = pd.concat([df_wiim, df_inyong], ignore_index=True)
    df_all = df_all[~df_all["법령_key"].isna()].copy()

    agg = (
        df_all.groupby("법령_key", dropna=True)
        .agg({
            "법령":"first",
            "시행령_arts":       lambda x: list(dict.fromkeys(chain.from_iterable(x.tolist()))),
            "시행규칙_arts":     lambda x: list(dict.fromkeys(chain.from_iterable(x.tolist()))),
            "시행령_blocks":     lambda x: merge_blocks_maxlen(x.tolist()),
            "시행규칙_blocks":   lambda x: merge_blocks_maxlen(x.tolist())
        })
        .reset_index()
    )

    rows = []
    for _, r in agg.iterrows():
        law_key   = r["법령_key"]
        law_text  = normalize_spaces(r["법령"])
        sye_arts  = r["시행령_arts"]       or []
        syk_arts  = r["시행규칙_arts"]     or []
        sye_blocks= r["시행령_blocks"]     or {}
        syk_blocks= r["시행규칙_blocks"]   or {}

        # same-num
        same_payload = []
        if law_key in sye_arts and law_key in sye_blocks:
            same_payload.append({"source_type":"시행령","article":law_key,"text":sye_blocks[law_key]})
        if law_key in syk_arts and law_key in syk_blocks:
            same_payload.append({"source_type":"시행규칙","article":law_key,"text":syk_blocks[law_key]})
        if same_payload:
            rows.append({
                "법령_key": law_key, "family_type":"same-num", "target_num": law_key,
                "본문_법령": law_text,
                "연결_컨텐츠": json.dumps(same_payload, ensure_ascii=False)
            })

        # cross-num
        cross_targets = sorted(set([*sye_arts, *syk_arts]) - {law_key}, key=key_num)
        for t in cross_targets:
            payload=[]
            if t in sye_blocks:
                payload.append({"source_type":"시행령","article":t,"text":sye_blocks[t]})
            if t in syk_blocks:
                payload.append({"source_type":"시행규칙","article":t,"text":syk_blocks[t]})
            if payload:
                rows.append({
                    "법령_key": law_key, "family_type":"cross-num", "target_num": t,
                    "본문_법령": law_text,
                    "연결_컨텐츠": json.dumps(payload, ensure_ascii=False)
                })

        # 링크 전혀 없을 때 solo
        if not same_payload and not cross_targets:
            rows.append({
                "법령_key": law_key, "family_type":"solo", "target_num": law_key,
                "본문_법령": law_text,
                "연결_컨텐츠": json.dumps([], ensure_ascii=False)
            })

    family_df = pd.DataFrame(rows)
    # 정렬
    def key_num2(s: str) -> int:
        m = re.search(r"\d+", s or "")
        return int(m.group(0)) if m else 10**9
    family_df["법령_num"]       = family_df["법령_key"].apply(key_num2)
    family_df["target_num_num"] = family_df["target_num"].apply(key_num2)
    family_df = family_df.sort_values(["법령_num","family_type","target_num_num"])\
                         .drop(columns=["법령_num","target_num_num"])\
                         .reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    family_df.to_csv(out_path,index=False)
    print(f"[OK] {out_path}")

def build_all_family_csvs(base_root: Path):
    law_short = {
        '개인정보보호법':'개인', '신용정보법':'신용', '자본시장법':'자본',
        '전자금융거래법':'금융', '전자서명법':'서명', '정보통신망법':'통신'
    }
    tri_dir = base_root / "3단비교"
    for law, short in law_short.items():
        root_dir = tri_dir / law
        build_family_csv(root_dir, short)

# ==============================
# (C) PDF → 조문 단위 docs 변환 (공통)
# ==============================
CHAPTER_PAT = re.compile(r"^\s*(제\s*\d+\s*장)\s*([^\n]*)", re.MULTILINE)
ARTICLE_PAT = re.compile(
    r"^\s*(제\s*\d+\s*조(?:\s*의\s*\d+)?)\s*(?:(?:\(([^)]+)\)|（([^）]+)）)|삭제|준용)",
    re.MULTILINE
)
ADDENDUM_PAT = re.compile(r"^\s*부\s*칙\s*(?:<([^>]+)>)?(?:\s*\(([^)]+)\))?", re.MULTILINE)

def pdf_to_docs_by_article(pdf_path: str) -> List[Document]:
    """
    공통: PDF 전체 텍스트에서 장/조문/부칙을 인식하여 조문 단위 Document 리스트 생성
    - 부칙 내부에 조문이 없을 때, 부칙 전체를 1개의 문서로 보존
    """
    pages = PyPDFLoader(pdf_path).load()
    full_text = "\n".join(p.page_content for p in pages) + "\n"

    chapters = [(m.start(), m.group(1).replace(" ", ""), (m.group(2) or "").strip())
                for m in CHAPTER_PAT.finditer(full_text)]
    articles = [(m.start(), m.group(1).replace(" ", ""), (m.group(2) or m.group(3) or "").strip())
                for m in ARTICLE_PAT.finditer(full_text)]
    articles_sorted = sorted(articles, key=lambda x: x[0])

    add_heads = [(m.start(), (m.group(1) or m.group(2) or "").strip())
                 for m in ADDENDUM_PAT.finditer(full_text)]
    addenda  = [(pos, "부칙", label) for (pos, label) in add_heads]
    chapters_all = sorted(chapters + addenda, key=lambda x: x[0])

    # 조문 span
    article_spans = []
    for i, (pos, art_no, art_title) in enumerate(articles_sorted):
        start = pos
        end   = articles_sorted[i+1][0] if i+1 < len(articles_sorted) else len(full_text)
        chapter_no, chapter_title = None, None
        for cpos, cno, ctitle in chapters_all:
            if cpos <= start:
                chapter_no, chapter_title = cno, ctitle
            else:
                break
        article_spans.append((start, end, art_no, art_title, chapter_no, chapter_title))

    docs_by_article = []
    for start, end, art_no, art_title, ch_no, ch_title in article_spans:
        text = full_text[start:end].strip()
        a_no = art_no if ch_no != "부칙" else f"부칙{art_no}"
        metadata = {
            "source": pdf_path,
            "chapter_no": ch_no,
            "chapter_title": ch_title,
            "article_no": a_no,
            "article_title": art_title,
            "granularity": "article",
            "is_addendum": (ch_no == "부칙"),
        }
        docs_by_article.append(Document(page_content=text, metadata=metadata))

    # 부칙 블록 전체 1개 문서 (부칙에 조문이 전혀 없을 때)
    add_bounds = []
    for i, (apos, label) in enumerate(add_heads):
        astart = apos
        aend   = add_heads[i+1][0] if i+1 < len(add_heads) else len(full_text)
        add_bounds.append((astart, aend, label))

    article_positions = [pos for (pos, _, _) in articles_sorted]
    for astart, aend, label in add_bounds:
        has_article_inside = any(astart <= p < aend for p in article_positions)
        if not has_article_inside:
            seg = full_text[astart:aend].strip()
            if seg:
                tag = _label_tag(label)
                metadata = {
                    "source": pdf_path,
                    "chapter_no": "부칙",
                    "chapter_title": label,
                    "article_no": f"부칙본문<{tag}>",
                    "article_title": "부칙 본문",
                    "granularity": "article",
                    "is_addendum": True,
                }
                docs_by_article.append(Document(page_content=seg, metadata=metadata))

    return docs_by_article

# ==============================
# (D) 인덱싱(ED/ER/REG/BASE) 빌더 (공통)
# ==============================
def _family_from_name_generic(name: str) -> str:
    if "전자금융거래" in name or "전자금융" in name: return "전자금융"
    if "정보통신망" in name:   return "정보통신망"
    if "신용정보"   in name:   return "신용정보"
    if "전자서명"   in name:   return "전자서명"
    if "개인정보보호" in name: return "개인정보보호"
    if "자본시장"   in name:   return "자본시장"
    return name

def _index_from_pdf_patterns(
    in_dir: Path,
    out_dir: Path,
    targets: Dict[str, str],
    corpus_type: str,
    family_name_override=None,
    device_for_embed: str = "cpu",
    embed_model: str = "BAAI/bge-m3"
):
    """
    - in_dir: PDF 폴더
    - out_dir: 인덱스 저장 루트
    - targets: {표시이름: glob패턴 또는 파일명}
    - corpus_type: '시행령'/'시행규칙'/'감독규정'/'운영기준' 등
    - family_name_override: callable(name) -> family (없으면 _family_from_name_generic 사용)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = HuggingFaceBgeEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": device_for_embed},
        encode_kwargs={"normalize_embeddings": True}
    )

    for doc_name, pattern in targets.items():
        matches = sorted(glob.glob(str(in_dir / pattern)))
        if not matches:
            print(f"[SKIP] {doc_name}: {pattern} 매칭 파일 없음")
            continue

        pdf_path = matches[0]
        print(f"\n=== {doc_name} ===")
        print("PDF:", os.path.basename(pdf_path))

        docs_by_article = pdf_to_docs_by_article(pdf_path)

        fam = family_name_override(doc_name) if family_name_override else _family_from_name_generic(doc_name)

        patched = []
        for i, d in enumerate(docs_by_article):
            md = dict(d.metadata)
            md.update({
                "corpus_type": corpus_type,
                "law_full_name": doc_name,
                "law_family": fam,
                "doc_id": f"{doc_name}::{md.get('article_no','?')}::{i}",
            })
            patched.append(type(d)(page_content=d.page_content, metadata=md))

        store = FAISS.from_documents(patched, emb)
        save_dir = out_dir / doc_name
        store.save_local(str(save_dir))
        print("저장:", save_dir, "| 조문 수:", len(patched))

# ==============================
# (E) FAMILY 통합 인덱스 (CSV 기반) 생성
# ==============================
LAW_SHORT = {
    '개인정보보호법':'개인',
    '신용정보법':'신용',
    '자본시장법':'자본',
    '전자금융거래법':'금융',
    '전자서명법':'서명',
    '정보통신망법':'통신',
}

def _norm_article_no_simple(s: str) -> str:
    """
    '제10조', '제10조의2' 같은 첫 매치 문자열을 반환.
    ART_PAT이 캡처그룹이 없을 수도 있으니 group(0)/group(1) 모두 대응.
    """
    s = re.sub(r"\s+", "", s or "")
    m = ART_PAT.search(s)
    if not m:
        return ""
    # m.lastindex가 있으면 캡처그룹이 있다는 뜻
    return m.group(1) if (m.lastindex and m.lastindex >= 1) else m.group(0)

def family_csv_path(family_root: Path, law_folder: str) -> str:
    short = LAW_SHORT.get(law_folder)
    if not short:
        raise ValueError(f"LAW_SHORT에 '{law_folder}' 매핑 없음")
    return str(family_root / law_folder / f"{short}_family.csv")

@lru_cache(maxsize=16)
def load_family_df(family_root: Path, law_folder: str) -> pd.DataFrame:
    csv_path = family_csv_path(family_root, law_folder)
    df = pd.read_csv(csv_path)
    def _parse_json(x):
        try: return json.loads(x) if isinstance(x, str) and x.strip() else []
        except: return []
    if "연결_컨텐츠" in df.columns and "연결_컨텐츠_list" not in df.columns:
        df["연결_컨텐츠_list"] = df["연결_컨텐츠"].apply(_parse_json)
    elif "연결_컨텐츠_list" not in df.columns:
        df["연결_컨텐츠_list"] = [[] for _ in range(len(df))]
    return df

def build_family_all_index(
    output_dir: Path,
    family_root: Path,
    include_links: bool = True,
    embed_model: str = "BAAI/bge-m3",
    embed_device: str = "cpu",
    embed_batch: int = 8,
    add_batch: int = 256,
    force_rebuild: bool = False,
):
    if (not force_rebuild) and (
        (output_dir / "index.faiss").exists() or (output_dir / "index.pkl").exists()
    ):
        print(f"[SKIP] already exists: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    emb = HuggingFaceBgeEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": embed_device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": embed_batch}
    )

    docs: List[Document] = []

    # 조문별 법령 본문(가장 긴 것 1개) + (옵션) 연결 스니펫(시행령/시행규칙/감독규정)
    for law in LAW_SHORT.keys():
        csv_path = family_csv_path(family_root, law)
        if not os.path.exists(csv_path):
            print(f"[WARN] CSV not found: {csv_path}  (skip)")
            continue

        df = load_family_df(family_root, law)

        # (a) 법령 본문(조문별 가장 긴 버전)
        law_text_by_art: Dict[str,str] = {}
        for _, r in df.iterrows():
            art = _norm_article_no_simple(r.get("법령_key"))
            txt = (r.get("본문_법령") or "").strip()
            if not art or not txt:
                continue
            if art not in law_text_by_art or len(law_text_by_art[art]) < len(txt):
                law_text_by_art[art] = txt

        for art, txt in law_text_by_art.items():
            docs.append(Document(
                page_content=txt,
                metadata={"law_full_name": law, "article_no": art, "corpus_type": "법령", "source": f"{law}/family.csv"}
            ))

        # (b) 연결 스니펫
        if include_links:
            for _, r in df.iterrows():
                for p in (r.get("연결_컨텐츠_list") or []):
                    st = p.get("source_type")
                    if st not in ("시행령","시행규칙","감독규정"):
                        continue
                    art = (p.get("article") or "").strip()
                    txt = (p.get("text") or "").strip()
                    if not txt:
                        continue
                    docs.append(Document(
                        page_content=txt,
                        metadata={
                            "law_full_name": f"{law}{st}",
                            "article_no": art,
                            "corpus_type": ("감독규정" if st == "감독규정" else st),
                            "source": f"{law}/family.csv",
                        }
                    ))

    # 배치 add
    def _chunks(it, n):
        it = iter(it)
        while True:
            batch = list(islice(it, n))
            if not batch: break
            yield batch

    store = None
    first = True
    for batch in _chunks(docs, add_batch):
        if first:
            store = FAISS.from_documents(batch, emb)
            first = False
        else:
            store.add_documents(batch)

    if store is None:
        raise RuntimeError("No documents to index for family_all_index")
    store.save_local(str(output_dir))
    print(f"[OK] saved family index: {output_dir}  (docs={len(docs)})")

# ==============================
# (F) 인덱스 품질 진단 (Audit)
# ==============================
def _is_faiss_dir(p: str) -> bool:
    return os.path.isfile(os.path.join(p, "index.faiss")) or os.path.isfile(os.path.join(p, "index.pkl"))

def _all_faiss_dirs(root: str):
    if not root or not os.path.exists(root): return []
    if _is_faiss_dir(root): return [root]
    return [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d) and _is_faiss_dir(d)]

ART_RE = re.compile(r"^(?:부칙)?제\d+조(?:의\d+)?(?:제\d+항)?$")
def _norm(s: str|None) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s)

def _short(txt: str, n=140) -> str:
    return textwrap.shorten(" ".join((txt or "").split()), width=n, placeholder=" …")

def _hash(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

def _summarize_counts(arr):
    if not arr: return (0, 0, 0)
    arr = sorted(arr)
    return (arr[0], arr[len(arr)//2], arr[-1])

def _dociter(store):
    dct = getattr(store.docstore, "_dict", {}) or {}
    return list(dct.values())

def audit_index_dir(path: str, label: str, emb_for_load: HuggingFaceEmbeddings):
    try:
        store = FAISS.load_local(path, emb_for_load, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"[FAIL] load: {path} -> {e}")
        return

    docs = _dociter(store)
    n = len(docs)
    if n == 0:
        print(f"[{label}] {path}: empty.")
        return

    missing_article = 0
    weird_article = 0
    gran = Counter()
    by_article = defaultdict(list)
    lengths = []
    dup_by_art = defaultdict(set)
    src_names = Counter()

    for d in docs:
        md = d.metadata or {}
        art = _norm(md.get("article_no"))
        gran[md.get("granularity","")] += 1
        src_names[md.get("law_full_name") or md.get("source") or md.get("corpus_type") or "unknown"] += 1

        if not art:
            missing_article += 1
        else:
            if not ART_RE.match(art):
                weird_article += 1
            by_article[art].append(d)

        lengths.append(len((d.page_content or "")))
        dup_by_art[art].add(_hash(d.page_content or ""))

    num_articles = len(by_article)
    docs_per_art = [len(v) for v in by_article.values()]
    min_k, med_k, max_k = _summarize_counts(docs_per_art)
    L = (np.percentile(lengths, [5, 25, 50, 75, 95]).astype(int).tolist()
         if lengths else [0]*5)

    print(f"\n=== [{label}] {path} ===")
    print(f"- 총 문서수: {n:,}")
    print(f"- 고유 조문수: {num_articles:,}")
    print(f"- article_no 누락: {missing_article:,} / 이상패턴: {weird_article:,}")
    print(f"- granularity 분포: {dict(gran)}")
    print(f"- 조문당 문서수(최소/중간/최대): {min_k}/{med_k}/{max_k}")
    print(f"- 컨텐츠 길이 분위수(chars) p5/p25/p50/p75/p95: {L}")
    if num_articles:
        print(f"- 소스/법령명 상위: {src_names.most_common(3)}")
        print("\n-- 샘플(각 조문 1개씩) --")
        cnt = 0
        for art, arr in sorted(by_article.items())[:3]:
            d0 = arr[0]
            name = (d0.metadata or {}).get("law_full_name") or (d0.metadata or {}).get("source") or (d0.metadata or {}).get("corpus_type") or "문서"
            title = (d0.metadata or {}).get("article_title") or ""
            print(f"  [{name}] {art} {title}")
            print("   ", _short(d0.page_content))
            cnt += 1
        if cnt == 0:
            print("  (조문 샘플 없음)")

        heavy_dups = [(art, len(s)) for art, s in dup_by_art.items() if len(s) < len(by_article[art])]
        if heavy_dups:
            heavy_dups.sort(key=lambda x: (len(by_article[x[0]]) - x[1]), reverse=True)
            print("\n-- 텍스트 중복 의심(동일해시 다수) 상위 --")
            for art, uniq in heavy_dups[:5]:
                total = len(by_article[art])
                print(f"  {art}: {uniq} unique / {total} docs")

def audit_root(root: Path, label: str, emb_for_load: HuggingFaceEmbeddings):
    paths = _all_faiss_dirs(str(root))
    if not paths:
        print(f"[WARN] {label}: 인덱스 폴더 없음 or 비어있음 -> {root}")
        return
    for p in paths:
        audit_index_dir(p, label, emb_for_load)

# ==============================
# (G) RAG / Warmup / Inference 유틸 (inference.py에서 import)
# ==============================
# 전역(로드 관련)
FAMILY_INDEX_ROOT: Path = Path("./index_family")
DECREE_INDEX_ROOT: Path = Path("./index_ED")
RULE_INDEX_ROOT:   Path = Path("./index_ER")
REG_INDEX_ROOT:    Path = Path("./index_REG")
OPER_INDEX_ROOT:   Path = Path("./index_BASE")

FAMILY_ROOT:       Path = Path("./3단비교")

INSTR = "Represent this sentence for searching relevant passages: "
_emb_query = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True, "batch_size": 8})

FAM_STORE = None
DECREE_STORES = []
RULE_STORES = []
REG_STORES = []
OPER_STORES = []
_warmed_up = False

# 질문 타입 힌트 탐지
LAW_NAME_PATTERNS = {
    '개인정보보호법': re.compile(r"(개인정보\s*보호법|개보법)", re.I),
    '신용정보법':     re.compile(r"(신용정보의\s*이용\s*및\s*보호에\s*관한\s*법률|신용정보법)", re.I),
    '자본시장법':     re.compile(r"(자본\s*시장\s*법|자본시장과\s*금융투자업에\s*관한\s*법률|자본시장법)", re.I),
    '전자금융거래법': re.compile(r"(전자\s*금융\s*거래\s*법|전금법)", re.I),
    '전자서명법':     re.compile(r"(전자\s*서명\s*법)", re.I),
    '정보통신망법':   re.compile(r"(정보통신망\s*(이용촉진\s*및\s*정보보호\s*등에\s*관한\s*법률|법)|정통망법|망법)", re.I),
}
FAMILY_KEYWORDS = {
    '신용정보법':     [r'개인신용정보', r'신용정보(?!원장)'],
    '전자금융거래법': [r'전자금융'],
    '개인정보보호법': [r'개인정보(?!보호위원회)'],
    '자본시장법':     [r'금융투자|집합투자'],
    '전자서명법':     [r'전자서명|인증서'],
    '정보통신망법':   [r'정보통신망'],
}

def _nz(x): return "" if x is None else str(x).strip()

def _is_faiss_dir_p(p: Path) -> bool:
    return (p / "index.faiss").exists() or (p / "index.pkl").exists()

def _all_faiss_dirs_p(root: Path) -> List[Path]:
    if not root.exists(): return []
    if _is_faiss_dir_p(root): return [root]
    return [p for p in root.iterdir() if p.is_dir() and _is_faiss_dir_p(p)]

def _guess_law_name(md: Dict[str, Any]):
    return _nz(md.get("law_full_name") or md.get("corpus_type") or md.get("source") or "문서")

def build_ctx_text(docs: List[Any], char_budget: int = 2000, per_doc_min: int = 220) -> str:
    blocks, used = [], 0
    per_doc = max(per_doc_min, (char_budget // max(1, len(docs))) - 40)
    for i, d in enumerate(docs, 1):
        md = getattr(d, "metadata", None) or {}
        body = getattr(d, "page_content", "") or ""
        head = " ".join(p for p in [f"[{i}]", f"[{_guess_law_name(md)}]", _nz(md.get("chapter_no")), _nz(md.get("article_no")), _nz(md.get("article_title"))] if p).strip()
        snippet = body if len(body) <= per_doc else body[:per_doc] + " …"
        block = f"{head}\n{snippet}"
        if used + len(block) > char_budget: break
        blocks.append(block); used += len(block)
    return "\n\n".join(blocks)

def _load_named_stores(root: Path, emb_for_load: HuggingFaceEmbeddings):
    stores = []
    for p in _all_faiss_dirs_p(root):
        name = p.name
        try:
            st = FAISS.load_local(str(p), emb_for_load, allow_dangerous_deserialization=True)
            stores.append((name, st))
        except Exception as e:
            print(f"[WARN] fail load {p}: {e}")
    return stores

def warmup_retrievers(cache_decrees=True, cache_rules=True, cache_regs=True, cache_oper=True):
    global FAM_STORE, DECREE_STORES, RULE_STORES, REG_STORES, OPER_STORES, _warmed_up
    if _warmed_up:
        return
    if not _is_faiss_dir_p(FAMILY_INDEX_ROOT):
        raise RuntimeError(f"family index not found: {FAMILY_INDEX_ROOT}")
    FAM_STORE      = FAISS.load_local(str(FAMILY_INDEX_ROOT), _emb_query, allow_dangerous_deserialization=True)
    DECREE_STORES  = _load_named_stores(DECREE_INDEX_ROOT, _emb_query) if cache_decrees else []
    RULE_STORES    = _load_named_stores(RULE_INDEX_ROOT,   _emb_query) if cache_rules   else []
    REG_STORES     = _load_named_stores(REG_INDEX_ROOT,    _emb_query) if cache_regs    else []
    OPER_STORES    = _load_named_stores(OPER_INDEX_ROOT,   _emb_query) if cache_oper    else []
    _warmed_up = True

def warmup_from_manifest(manifest_path: str, cache_decrees=True, cache_rules=True, cache_regs=True, cache_oper=True):
    """
    inference.py에서 호출:
        - manifest.json을 읽어 각 인덱스 루트를 갱신
        - retriever들을 로드
    """
    global FAMILY_INDEX_ROOT, DECREE_INDEX_ROOT, RULE_INDEX_ROOT, REG_INDEX_ROOT, OPER_INDEX_ROOT, FAMILY_ROOT, _warmed_up
    mp = Path(manifest_path).resolve()
    if not mp.exists():
        raise FileNotFoundError(f"manifest not found: {mp}")
    data = json.loads(mp.read_text(encoding="utf-8"))
    FAMILY_INDEX_ROOT = Path(data.get("law_index", "./index_family")).resolve()
    DECREE_INDEX_ROOT = Path(data.get("decree_index", "./index_ED")).resolve()
    RULE_INDEX_ROOT   = Path(data.get("rule_index", "./index_ER")).resolve()
    REG_INDEX_ROOT    = Path(data.get("reg_index", "./index_REG")).resolve()
    OPER_INDEX_ROOT   = Path(data.get("oper_index", "./index_BASE")).resolve()
    # family_root는 manifest에 없으니 기본값 유지
    _warmed_up = False
    warmup_retrievers(cache_decrees=cache_decrees, cache_rules=cache_rules, cache_regs=cache_regs, cache_oper=cache_oper)

def extract_article_targets(text: str) -> list[str]:
    s = re.sub(r"\s+", "", text)
    pat = re.compile(r"제(\d+)조(?:의(\d+))?")
    out = []
    for m in pat.finditer(s):
        base, sub = m.group(1), m.group(2)
        out.append(f"제{base}조" + (f"의{sub}" if sub else ""))
    seen=set(); res=[]
    for t in out:
        if t not in seen:
            res.append(t); seen.add(t)
    return res

def extract_explicit_law_articles(text: str) -> list[tuple[str,str]]:
    pairs, arts = [], extract_article_targets(text)
    if not arts:
        return []
    for law, pat in LAW_NAME_PATTERNS.items():
        if pat.search(text):
            for art in arts:
                pairs.append((law, art))
    seen=set(); res=[]
    for p in pairs:
        if p not in seen:
            res.append(p); seen.add(p)
    return res

def build_query_for_retrieval(q: str) -> str:
    try:
        if is_multiple_choice(q):
            question, _, _, options = extract_question_and_choices(q)
            return question + "\n" + "\n".join(options)
    except Exception:
        pass
    return q

def detect_law_hint(text: str) -> str | None:
    for law, pat in LAW_NAME_PATTERNS.items():
        if pat.search(text):
            return law
    for law, pats in FAMILY_KEYWORDS.items():
        if any(re.search(p, text) for p in pats):
            return law
    return None

def _detect_doc_type_hint(text: str) -> str | None:
    s = text.lower()
    if '감독규정' in s: return '감독규정'
    if re.search(r'운영\s*기준', s): return '운영기준'
    if '시행령'   in s: return '시행령'
    if '시행규칙' in s: return '시행규칙'
    return None

class MiniDoc:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

def get_family_all_docs(family_root: Path, law_folder: str, target_no: str) -> List[MiniDoc]:
    df = load_family_df(family_root, law_folder)
    tgt = _norm_article_no_simple(target_no)
    sub = df[df["법령_key"].map(_norm_article_no_simple) == tgt].copy()

    docs: List[MiniDoc] = []
    if not sub.empty:
        best = max(((r.get("본문_법령") or "") for _, r in sub.iterrows()), key=lambda s: len(s or ""), default="")
        if best:
            docs.append(MiniDoc(best, {"law_full_name": law_folder, "article_no": tgt, "corpus_type": "법령"}))

    for family_type in ("same-num", "cross-num"):
        for _, r in sub[sub["family_type"] == family_type].iterrows():
            for p in (r.get("연결_컨텐츠_list") or []):
                st = p.get("source_type"); art=p.get("article") or ""; txt=(p.get("text") or "").strip()
                if not txt or st not in ("시행령","시행규칙","감독규정"):
                    continue
                docs.append(MiniDoc(txt, {
                    "law_full_name": f"{law_folder}{st}",
                    "article_no": art,
                    "corpus_type": ("감독규정" if st == "감독규정" else st),
                    "family_type": family_type,
                    "family_target": r.get("target_num","")
                }))
    return docs

def _scope_named_stores(named_stores, law_hint: str|None):
    if not law_hint:
        return []
    family_prefix = '신용정보' if '신용정보' in law_hint else \
                    '전자금융' if '전자금융' in law_hint else \
                    '개인정보보호' if '개인정보보호' in law_hint else \
                    '자본시장' if '자본시장' in law_hint else \
                    '전자서명' if '전자서명' in law_hint else \
                    '정보통신망' if '정보통신망' in law_hint else ''
    if not family_prefix:
        family_prefix = law_hint

    out = []
    for (name, st) in named_stores:
        base = (name or '').replace(' ', '')
        if base.startswith(family_prefix) or (family_prefix in base):
            out.append((name, st))
    return out

def retrieve_and_rerank_with_vec(
        q_vec,
        question_text: str,
        top_k_family: int = 6,
        top_k_decree_per_store: int = 2,
        final_top_n: int = 8,
        law_hint: str | None = None):

    assert FAM_STORE is not None, "warmup_retrievers() 먼저 호출하세요."
    results = []
    doc_type_hint = _detect_doc_type_hint(question_text)

    # family
    try:
        fam_hits = FAM_STORE.similarity_search_by_vector_with_relevance_scores(q_vec, k=top_k_family)
    except AttributeError:
        docs = FAM_STORE.similarity_search_by_vector(q_vec, k=top_k_family) or []
        fam_hits = [(d, 0.0) for d in docs]
    for d, rel in fam_hits:
        md = dict(d.metadata or {})
        md.setdefault("corpus_type", "법령")
        d.metadata = md
        results.append((d, float(rel)))

    # 시행령
    for (name, store) in _scope_named_stores(DECREE_STORES, law_hint):
        try:
            hits = store.similarity_search_by_vector_with_relevance_scores(q_vec, k=top_k_decree_per_store)
        except AttributeError:
            docs = store.similarity_search_by_vector(q_vec, k=top_k_decree_per_store) or []
            hits = [(d, 0.0) for d in docs]
        for d, rel in hits:
            md = dict(d.metadata or {}); md["corpus_type"] = "시행령"; d.metadata = md
            results.append((d, float(rel)))

    # 시행규칙
    for (name, store) in _scope_named_stores(RULE_STORES, law_hint):
        try:
            hits = store.similarity_search_by_vector_with_relevance_scores(q_vec, k=top_k_decree_per_store)
        except AttributeError:
            docs = store.similarity_search_by_vector(q_vec, k=top_k_decree_per_store) or []
            hits = [(d, 0.0) for d in docs]
        for d, rel in hits:
            md = dict(d.metadata or {}); md["corpus_type"] = "시행규칙"; d.metadata = md
            results.append((d, float(rel)))

    # 감독규정
    reg_scope = REG_STORES if (doc_type_hint == '감독규정' or (law_hint and '전자금융' in law_hint)) \
                            else _scope_named_stores(REG_STORES, law_hint)
    for (name, store) in reg_scope:
        try:
            hits = store.similarity_search_by_vector_with_relevance_scores(q_vec, k=top_k_decree_per_store)
        except AttributeError:
            docs = store.similarity_search_by_vector(q_vec, k=top_k_decree_per_store) or []
            hits = [(d, 0.0) for d in docs]
        for d, rel in hits:
            md = dict(d.metadata or {}); md["corpus_type"] = "감독규정"; d.metadata = md
            results.append((d, float(rel)))

    # 운영기준(전자서명인증업무운영기준)
    oper_scope = OPER_STORES if doc_type_hint == '운영기준' else _scope_named_stores(OPER_STORES, law_hint)
    for (name, store) in oper_scope:
        try:
            hits = store.similarity_search_by_vector_with_relevance_scores(q_vec, k=top_k_decree_per_store)
        except AttributeError:
            docs = store.similarity_search_by_vector(q_vec, k=top_k_decree_per_store) or []
            hits = [(d, 0.0) for d in docs]
        for d, rel in hits:
            md = dict(d.metadata or {}); md["corpus_type"] = "운영기준"; d.metadata = md
            results.append((d, float(rel)))

    if not results:
        return []

    raw = np.array([s for _, s in results], dtype=float)
    denom = (raw.max() - raw.min()) or 1.0
    sim01 = (raw - raw.min()) / denom

    reranked = []
    for (doc, _s), s01 in zip(results, sim01):
        name  = (doc.metadata or {}).get("law_full_name", "")
        ctype = (doc.metadata or {}).get("corpus_type", "")
        score = 0.75 * s01

        if law_hint:
            fam_hit = (
                (name.startswith('전자서명')   and '전자서명'   in law_hint) or
                (name.startswith('신용정보')   and '신용정보'   in law_hint) or
                (name.startswith('전자금융')   and '전자금융'   in law_hint) or
                (name.startswith('정보통신망') and '정보통신망' in law_hint) or
                (name.startswith('개인정보보호') and '개인정보보호' in law_hint) or
                (name.startswith('자본시장')   and '자본시장'   in law_hint)
            )
            if fam_hit: score += 0.30
            else:       score -= 0.50

        if _detect_doc_type_hint(question_text) == ctype:
            score += 0.60
        else:
            score -= 0.20

        reranked.append((score, doc))

    reranked.sort(key=lambda x: x[0], reverse=True)

    picked, seen = [], set()
    for _, d in reranked:
        key = (
            (d.metadata or {}).get("law_full_name"),
            (d.metadata or {}).get("article_no"),
            (d.metadata or {}).get("corpus_type"),
        )
        if key in seen:
            continue
        seen.add(key)
        picked.append(d)
        if len(picked) >= final_top_n:
            break
    return picked

# 객관식 판별/파싱 (inference.py에서 사용)
def is_multiple_choice(question_text):
    lines = (question_text or "").strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text):
    lines = (full_text or "").strip().split("\n")
    q_lines = []; num_list = []; content_list = []; options = []
    for line in lines:
        m = re.match(r"^\s*([1-9][0-9]?)\s+(.*\S)\s*$", line)
        if m:
            num, content = m.group(1), m.group(2)
            num_list.append(num); content_list.append(content); options.append(f"{num} {content}")
        else:
            q_lines.append(line.strip())
    question = " ".join(q_lines)
    return question, num_list, content_list, options

def build_context_for_question(q: str, char_budget: int = 2000) -> str:
    doc_type_hint = _detect_doc_type_hint(q)

    pairs = extract_explicit_law_articles(q)
    if pairs and doc_type_hint not in ('감독규정', '운영기준'):
        docs=[]
        for law, art in pairs:
            docs.extend(get_family_all_docs(FAMILY_ROOT, law, art))
        return build_ctx_text(docs[:10], char_budget=char_budget)

    law_hint = detect_law_hint(q)
    if not is_multiple_choice(q):
        if not law_hint:
            return ""

    query = build_query_for_retrieval(q)
    q_vec = _emb_query.embed_query(INSTR + query)
    top_docs = retrieve_and_rerank_with_vec(
        q_vec, query,
        top_k_family=6, top_k_decree_per_store=2, final_top_n=8,
        law_hint=law_hint
    )
    return build_ctx_text(top_docs, char_budget=char_budget)

def add_ctx_to_prompt(user_prompt: str, ctx_text: str | None = None) -> str:
    if ctx_text and ctx_text.strip():
        return f"[참고할 문서]\n{ctx_text}\n\n\n{user_prompt}"
    return user_prompt

def extract_answer_only(generated_text: str, original_question: str) -> str:
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()
    if not text:
        return "미응답"
    if is_multiple_choice(original_question):
        m = re.match(r"\D*([1-9][0-9]?)", text)
        return m.group(1) if m else "0"
    return text

# ==============================
# (H) main() : 전체 파이프라인 (빌드/인덱싱/감사/manifest)
# ==============================
def main(base_root: str = "./"):
    set_seed(208)

    BASE_ROOT = Path(base_root).resolve()

    # 경로들
    TRI_DIR      = BASE_ROOT / "3단비교"
    ED_DIR       = BASE_ROOT / "ED"
    ER_DIR       = BASE_ROOT / "ER"
    REG_DIR      = BASE_ROOT / "reg"
    BASE_DIR     = BASE_ROOT / "base"

    INDEX_FAMILY = BASE_ROOT / "index_family"
    INDEX_ED     = BASE_ROOT / "index_ED"
    INDEX_ER     = BASE_ROOT / "index_ER"
    INDEX_REG    = BASE_ROOT / "index_REG"
    INDEX_BASE   = BASE_ROOT / "index_BASE"
    ARTIFACTS    = BASE_ROOT / "artifacts"

    _ensure_dirs(TRI_DIR, ED_DIR, ER_DIR, REG_DIR, BASE_DIR,
                 INDEX_FAMILY, INDEX_ED, INDEX_ER, INDEX_REG, INDEX_BASE, ARTIFACTS)

    # 1) 3단비교 HTML→CSV
    build_3dan_csvs(BASE_ROOT)

    # 2) FAMILY CSV 생성
    build_all_family_csvs(BASE_ROOT)

    # 3) ED/ER/REG/BASE PDF 인덱스 생성
    # ED (시행령)
    ed_targets = {
        "전자금융거래법시행령": "*전자금융거래법*시행령*.pdf",
        "정보통신망법시행령":   "*정보통신망*법*시행령*.pdf",
        "신용정보법시행령":     "*신용정보*법*시행령*.pdf",
        "전자서명법시행령":     "*전자서명*법*시행령*.pdf",
        "개인정보보호법시행령": "*개인정보보호법*시행령*.pdf",
        "자본시장보호법시행령": "*자본시장보호법*시행령*.pdf"
    }
    _index_from_pdf_patterns(
        in_dir=ED_DIR, out_dir=INDEX_ED, targets=ed_targets, corpus_type="시행령",
        family_name_override=_family_from_name_generic, device_for_embed="cpu"
    )

    # ER (시행규칙)
    er_targets = {
        "정보통신망법시행규칙":   "*정보통신망*법*시행규칙*.pdf",
        "신용정보법시행규칙":     "*신용정보*법*시행규칙*.pdf",
        "전자서명법시행규칙":     "*전자서명*법*시행규칙*.pdf",
        "자본시장보호법시행규칙": "*자본시장보호법*시행규칙*.pdf"
    }
    _index_from_pdf_patterns(
        in_dir=ER_DIR, out_dir=INDEX_ER, targets=er_targets, corpus_type="시행규칙",
        family_name_override=_family_from_name_generic, device_for_embed="cpu"
    )

    # REG (감독규정) - 신용정보업감독규정, 전자금융감독규정
    reg_targets_credit = {"신용정보업감독규정": "신용정보업감독규정.pdf"}
    _index_from_pdf_patterns(
        in_dir=REG_DIR, out_dir=INDEX_REG, targets=reg_targets_credit, corpus_type="감독규정",
        family_name_override=lambda n: "신용정보", device_for_embed="cpu"
    )
    reg_targets_ef = {"전자금융감독규정": "전자금융감독규정.pdf"}
    _index_from_pdf_patterns(
        in_dir=REG_DIR, out_dir=INDEX_REG, targets=reg_targets_ef, corpus_type="감독규정",
        family_name_override=lambda n: "전자금융", device_for_embed="cpu"
    )

    # BASE (운영기준) - 전자서명인증업무운영기준
    base_targets = {"전자서명인증업무운영기준": "전자서명인증업무운영기준.pdf"}
    _index_from_pdf_patterns(
        in_dir=BASE_DIR, out_dir=INDEX_BASE, targets=base_targets, corpus_type="운영기준",
        family_name_override=lambda n: "전자서명인증", device_for_embed="cpu"
    )

    # 4) FAMILY 통합 인덱스 생성 (CSV 기반)
    build_family_all_index(
        output_dir=INDEX_FAMILY,
        family_root=TRI_DIR,
        include_links=True,
        embed_model="BAAI/bge-m3",
        embed_device="cpu",
        embed_batch=8,
        add_batch=256,
        force_rebuild=True,   # 필요 시 False로
    )

    # 5) 품질 진단 (간단)
    emb_for_load = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    audit_root(INDEX_FAMILY, "LAW", emb_for_load)
    audit_root(INDEX_ED,     "ED",  emb_for_load)
    audit_root(INDEX_ER,     "ER",  emb_for_load)
    audit_root(INDEX_REG,    "REG", emb_for_load)
    audit_root(INDEX_BASE,   "BASE",emb_for_load)

    # 6) manifest 저장
    manifest = {
        "embed_model": "BAAI/bge-m3",
        "law_index":    str(INDEX_FAMILY),
        "decree_index": str(INDEX_ED),
        "rule_index":   str(INDEX_ER),
        "reg_index":    str(INDEX_REG),
        "oper_index":   str(INDEX_BASE),
    }
    (ARTIFACTS / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("[OK] manifest saved:", ARTIFACTS / "manifest.json")

# ==============================
# Entry
# ==============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", type=str, default="./",
                        help="인덱스와 artifacts/manifest.json이 생성될 상위 폴더 (기본: 현재 폴더)")
    args = parser.parse_args()
    main(args.base_root)