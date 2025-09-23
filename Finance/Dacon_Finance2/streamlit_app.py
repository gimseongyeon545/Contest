import streamlit as st
import time
from inference import infer  # 🔁 추론 함수 inference.py

st.title("📘 금융 QA 응답기")
st.write("질문을 입력하면 모델이 답변을 생성합니다.")

# 입력 필드
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.session_state.user_input = st.text_area("질문 입력", value=st.session_state.user_input, height=150)

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("🔍 응답 생성")
with col2:
    clear = st.button("🧹 입력 초기화")

if run:
    if not st.session_state.user_input.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("모델이 응답을 생성 중입니다..."):
            try:
                start_time = time.time()  # 시작 시간
                st.session_state.result = infer(st.session_state.user_input)
                elapsed_time = time.time() - start_time  # 경과 시간
                st.session_state.elapsed_time = round(elapsed_time, 2)  # 소수점 2자리
            except Exception as e:
                st.error(f"❌ 에러 발생: {e}")

if clear:
    st.session_state.user_input = ""
    st.session_state.result = ""
    st.session_state.elapsed_time = 0
    st.rerun()

if "result" in st.session_state and st.session_state.result:
    st.success("✅ 생성된 답변:")
    st.write(st.session_state.result)

    if "elapsed_time" in st.session_state:
        st.info(f"⏱ 응답 생성 시간: {st.session_state.elapsed_time}초")
