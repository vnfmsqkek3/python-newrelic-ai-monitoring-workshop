"""
Streamlit + Bedrock + New Relic 모니터링 예제 (지연 버전)

모든 boto3.client('bedrock-runtime') 호출이 자동으로 모니터링되며,
의도적인 지연 함수들을 통해 성능 테스트 및 모니터링 검증이 가능합니다.
"""

import streamlit as st
import boto3
import json
import time
import uuid
import random
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# 상수 (라이브러리 import 전에 정의)
REGION = "ap-northeast-2"
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
APP_NAME = "gen-ai-bedrock-late"

# import nr_bedrock_observability 및 즉시 auto patch 활성화
import nr_bedrock_observability
nr_bedrock_observability.enable_auto_patch(application_name=APP_NAME)

from nr_bedrock_observability import (
    create_streamlit_evaluation_ui,
    create_streamlit_nrql_queries,
    get_streamlit_session_info,
    get_sample_nrql_queries
)

# 페이지 설정
st.set_page_config(
    page_title="Bedrock + New Relic (Late)", 
    page_icon="🐌",
    layout="wide"
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_role_prompt" not in st.session_state:
    st.session_state.user_role_prompt = "당신은 도움이 되는 AI 어시스턴트입니다."
if "user_system_prompt" not in st.session_state:
    st.session_state.user_system_prompt = "사용자의 질문에 명확하고 정확하게 답변해주세요. 한국어로 답변해주세요."
if "user_temperature" not in st.session_state:
    st.session_state.user_temperature = 0.7
if "user_top_p" not in st.session_state:
    st.session_state.user_top_p = 0.9
if "last_response_data" not in st.session_state:
    st.session_state.last_response_data = {}

# 추가 세션 상태 (모니터링용)
if "trace_id" not in st.session_state:
    st.session_state.trace_id = str(uuid.uuid4())
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "current_completion_id" not in st.session_state:
    st.session_state.current_completion_id = None
if "current_system_prompt" not in st.session_state:
    st.session_state.current_system_prompt = None
if "raw_result" not in st.session_state:
    st.session_state.raw_result = {}
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# 🐌 지연 설정 세션 상태
if "delay_enabled" not in st.session_state:
    st.session_state.delay_enabled = False
if "delay_type" not in st.session_state:
    st.session_state.delay_type = "simple_sleep"
if "delay_duration" not in st.session_state:
    st.session_state.delay_duration = 2.0
if "delay_before_llm" not in st.session_state:
    st.session_state.delay_before_llm = True
if "delay_after_llm" not in st.session_state:
    st.session_state.delay_after_llm = False

# 🐌 지연 함수들
def simple_sleep_delay(duration: float):
    """단순 Sleep 지연"""
    time.sleep(duration)

def cpu_intensive_delay(duration: float):
    """CPU 집약적 작업을 통한 지연"""
    start_time = time.time()
    target_time = start_time + duration
    
    # 의미없는 해시 계산으로 CPU 부하 생성
    counter = 0
    while time.time() < target_time:
        # SHA256 해시 계산
        data = f"delay_simulation_{counter}_{random.random()}".encode()
        hashlib.sha256(data).hexdigest()
        counter += 1
        
        # CPU 점유율 조절을 위한 짧은 휴식
        if counter % 1000 == 0:
            time.sleep(0.001)

def memory_intensive_delay(duration: float):
    """메모리 집약적 작업을 통한 지연"""
    start_time = time.time()
    target_time = start_time + duration
    
    memory_blocks = []
    try:
        while time.time() < target_time:
            # 큰 배열 생성 및 삭제
            block = np.random.rand(100000).astype(np.float64)  # ~800KB
            memory_blocks.append(block)
            
            # 메모리 정리 (일부만)
            if len(memory_blocks) > 10:
                memory_blocks.pop(0)
            
            time.sleep(0.1)  # 메모리 할당 간격
    finally:
        # 메모리 정리
        memory_blocks.clear()

def io_simulation_delay(duration: float):
    """I/O 작업 시뮬레이션을 통한 지연"""
    start_time = time.time()
    target_time = start_time + duration
    
    temp_files = []
    try:
        while time.time() < target_time:
            # 임시 파일 생성 및 쓰기
            temp_file = f"/tmp/delay_sim_{uuid.uuid4().hex[:8]}.txt"
            with open(temp_file, 'w') as f:
                for i in range(1000):
                    f.write(f"Line {i}: {random.random()}\n")
            
            temp_files.append(temp_file)
            
            # 파일 읽기
            with open(temp_file, 'r') as f:
                content = f.read()
            
            time.sleep(0.05)  # I/O 간격
            
            # 파일 개수 제한
            if len(temp_files) > 5:
                import os
                try:
                    os.remove(temp_files.pop(0))
                except:
                    pass
    finally:
        # 임시 파일 정리
        import os
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

def network_simulation_delay(duration: float):
    """네트워크 지연 시뮬레이션 (ThreadPoolExecutor 사용)"""
    start_time = time.time()
    target_time = start_time + duration
    
    def simulate_network_call():
        # 네트워크 호출 시뮬레이션
        time.sleep(random.uniform(0.1, 0.3))
        return f"Response_{random.random()}"
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        while time.time() < target_time:
            # 여러 개의 "네트워크 호출" 제출
            for _ in range(3):
                future = executor.submit(simulate_network_call)
                futures.append(future)
            
            # 일부 결과 기다리기
            if len(futures) >= 6:
                for future in futures[:3]:
                    try:
                        result = future.result(timeout=0.5)
                    except:
                        pass
                futures = futures[3:]
            
            time.sleep(0.2)

def mixed_delay(duration: float):
    """여러 지연 방법을 조합"""
    portion = duration / 4
    
    # 25%씩 각각 다른 방법 사용
    simple_sleep_delay(portion)
    cpu_intensive_delay(portion)
    memory_intensive_delay(portion)
    io_simulation_delay(portion)

# 지연 함수 매핑
DELAY_FUNCTIONS = {
    "simple_sleep": ("⏰ 단순 Sleep", simple_sleep_delay),
    "cpu_intensive": ("🔥 CPU 집약적", cpu_intensive_delay),
    "memory_intensive": ("💾 메모리 집약적", memory_intensive_delay),
    "io_simulation": ("📁 I/O 시뮬레이션", io_simulation_delay),
    "network_simulation": ("🌐 네트워크 시뮬레이션", network_simulation_delay),
    "mixed": ("🔀 혼합 방법", mixed_delay)
}

def apply_delay(stage: str):
    """지연 적용"""
    if not st.session_state.delay_enabled:
        return
    
    should_apply = False
    if stage == "before_llm" and st.session_state.delay_before_llm:
        should_apply = True
    elif stage == "after_llm" and st.session_state.delay_after_llm:
        should_apply = True
    
    if should_apply:
        delay_func_name, delay_func = DELAY_FUNCTIONS[st.session_state.delay_type]
        duration = st.session_state.delay_duration
        
        with st.spinner(f"🐌 {delay_func_name} 지연 중... ({duration}초)"):
            start_time = time.time()
            try:
                delay_func(duration)
                actual_duration = time.time() - start_time
                st.info(f"✅ 지연 완료: {actual_duration:.2f}초 ({stage})")
            except Exception as e:
                actual_duration = time.time() - start_time
                st.warning(f"⚠️ 지연 중 오류: {str(e)} (소요: {actual_duration:.2f}초)")

def get_bedrock_client():
    """자동 패치가 적용된 Bedrock 클라이언트 생성"""
    client = boto3.client('bedrock-runtime', region_name=REGION)
    return client 

# UI 구성
st.title("🐌 Auto Bedrock + New Relic (지연 테스트)")
st.caption("nr-bedrock-observability-python v2.4.1 - 자동 패치 모드 + 지연 시뮬레이션")

# 메인 레이아웃
main_col, sidebar_col = st.columns([2, 1])

with sidebar_col:
    st.header("⚙️ 설정")
    
    # 🐌 지연 설정 섹션
    st.subheader("🐌 지연 설정")
    
    # 지연 활성화/비활성화
    st.session_state.delay_enabled = st.checkbox(
        "지연 활성화", 
        value=st.session_state.delay_enabled,
        help="체크하면 의도적인 지연이 적용됩니다"
    )
    
    if st.session_state.delay_enabled:
        # 지연 타입 선택
        delay_options = list(DELAY_FUNCTIONS.keys())
        delay_labels = [DELAY_FUNCTIONS[key][0] for key in delay_options]
        
        selected_index = delay_options.index(st.session_state.delay_type)
        new_index = st.selectbox(
            "지연 방법",
            range(len(delay_options)),
            index=selected_index,
            format_func=lambda x: delay_labels[x],
            help="다양한 지연 방법을 선택할 수 있습니다"
        )
        st.session_state.delay_type = delay_options[new_index]
        
        # 지연 시간 설정
        st.session_state.delay_duration = st.slider(
            "지연 시간 (초)",
            min_value=0.5,
            max_value=10.0,
            value=st.session_state.delay_duration,
            step=0.5,
            help="지연할 시간을 초 단위로 설정"
        )
        
        # 지연 적용 시점
        st.markdown("**지연 적용 시점:**")
        st.session_state.delay_before_llm = st.checkbox(
            "🚀 LLM 호출 전", 
            value=st.session_state.delay_before_llm,
            help="Bedrock 호출 전에 지연 적용"
        )
        st.session_state.delay_after_llm = st.checkbox(
            "📝 LLM 응답 후", 
            value=st.session_state.delay_after_llm,
            help="Bedrock 응답 후에 지연 적용"
        )
        
        # 지연 설정 미리보기
        with st.expander("🔍 지연 설정 미리보기", expanded=False):
            delay_name, _ = DELAY_FUNCTIONS[st.session_state.delay_type]
            st.markdown(f"**방법:** {delay_name}")
            st.markdown(f"**시간:** {st.session_state.delay_duration}초")
            
            stages = []
            if st.session_state.delay_before_llm:
                stages.append("LLM 호출 전")
            if st.session_state.delay_after_llm:
                stages.append("LLM 응답 후")
            
            if stages:
                st.markdown(f"**적용 시점:** {', '.join(stages)}")
                total_delay = st.session_state.delay_duration * len(stages)
                st.markdown(f"**예상 총 지연:** {total_delay}초")
            else:
                st.warning("지연 적용 시점을 선택해주세요")
    
    st.markdown("---")
    
    # 모델 파라미터 설정
    st.header("모델 파라미터 설정")
    
    # 프리셋 버튼들
    st.subheader("빠른 설정 프리셋")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏥 의료 전문가", help="의료 관련 질문에 특화된 설정"):
            st.session_state.user_role_prompt = "당신은 15년 경력의 임상 의사이자 내과 전문의입니다. 환자들에게 실용적이고 구체적인 의학적 조언을 제공하는 것이 당신의 전문 분야입니다."
            st.session_state.user_system_prompt = """다음 원칙에 따라 의학적 조언을 제공하세요:

1. **전문적 진단과 치료 권고**: 증상을 분석하고 가능한 원인을 제시하며, 일반의약품이나 생활습관 개선 등 구체적인 해결책을 제안하세요.

2. **단계별 치료 접근**: 경증의 경우 자가 관리 방법부터 시작하여, 필요시 전문의 진료를 권하는 단계적 접근법을 사용하세요.

3. **실용적 약물 정보**: 일반의약품의 경우 구체적인 성분명, 용법·용량, 주의사항을 명확히 제시하세요.

4. **위험 신호 인식**: 응급 상황이나 전문의 진료가 반드시 필요한 경우를 명확히 구분하여 안내하세요.

5. **개인화된 조언**: 연령, 기존 질환, 복용 중인 약물 등을 고려한 맞춤형 조언을 제공하세요.

항상 의학적 근거를 바탕으로 하되, 환자가 실제로 활용할 수 있는 구체적이고 실용적인 정보를 우선 제공하세요. 한국어로 명확하고 이해하기 쉽게 답변해주세요."""
            st.session_state.user_temperature = 0.2
            st.session_state.user_top_p = 0.8
            st.rerun()
    
    with col2:
        if st.button("💻 코딩 튜터", help="프로그래밍 학습에 특화된 설정"):
            st.session_state.user_role_prompt = "당신은 10년 이상의 실무 경험을 가진 시니어 개발자이자 프로그래밍 교육 전문가입니다. 다양한 프로그래밍 언어와 프레임워크에 능통하며, 복잡한 개념을 쉽게 설명하는 것이 특기입니다."
            st.session_state.user_system_prompt = """다음 교육 방법론에 따라 프로그래밍 지도를 해주세요:

1. **단계별 학습 접근**:
   - 개념 설명 → 간단한 예제 → 실습 문제 → 심화 응용 순서로 진행
   - 학습자의 수준에 맞는 적절한 난이도 조절

2. **실습 중심 교육**:
   - 모든 개념에 대해 실행 가능한 코드 예제 제공
   - 주석을 상세히 달아 코드의 각 부분 설명
   - 실제 프로젝트에서 사용할 수 있는 실용적인 예제 우선

3. **디버깅과 문제 해결**:
   - 자주 발생하는 오류와 해결 방법 설명
   - 코드 리뷰 관점에서 개선점 제시
   - 효율적인 디버깅 방법 안내

4. **모범 사례와 코딩 표준**:
   - 클린 코드 작성 원칙 적용
   - 업계 표준과 모범 사례 소개
   - 성능 최적화와 유지보수성 고려

5. **학습 동기 부여**:
   - 실무에서의 활용 사례 제시
   - 단계별 성취감 제공
   - 추가 학습 자료와 발전 방향 제안

항상 '왜 이렇게 작성하는가?'에 대한 이유를 설명하고, 대안적 접근법도 함께 제시하세요. 한국어로 친근하고 이해하기 쉽게 설명해주세요."""
            st.session_state.user_temperature = 0.4
            st.session_state.user_top_p = 0.9
            st.rerun()
    
    with col3:
        if st.button("🎨 창의적 작가", help="창작 활동에 특화된 설정"):
            st.session_state.user_role_prompt = "당신은 베스트셀러 작품을 여러 편 출간한 전문 작가이자 창작 지도 전문가입니다. 소설, 에세이, 시나리오 등 다양한 장르에 능통하며, 독자의 마음을 사로잡는 스토리텔링이 특기입니다."
            st.session_state.user_system_prompt = """다음 창작 원칙에 따라 글쓰기 지도를 해주세요:

1. **창작 프로세스 안내**:
   - 아이디어 발굴 → 구성 계획 → 초고 작성 → 수정/편집 단계별 가이드
   - 각 단계에서 사용할 수 있는 구체적인 기법과 도구 제시

2. **스토리텔링 구조**:
   - 매력적인 도입부, 긴장감 있는 전개, 만족스러운 결말 구성법
   - 캐릭터 개발과 갈등 구조 설계
   - 장르별 특성을 고려한 맞춤형 조언

3. **문체와 표현력**:
   - 상황과 감정에 맞는 생생한 묘사 기법
   - 독자의 공감을 이끌어내는 표현 방법
   - 문장 리듬과 호흡을 고려한 글쓰기

4. **독창성과 차별화**:
   - 기존 작품과 차별화되는 독특한 관점 제시
   - 개인적 경험과 상상력을 결합한 오리지널 아이디어 개발
   - 트렌드를 반영하면서도 시대를 초월하는 보편성 추구

5. **실용적 글쓰기 조언**:
   - 작가의 블록 극복 방법
   - 효과적인 자료 조사와 취재 방법
   - 출간과 독자 소통을 위한 실무적 조언

항상 구체적인 예시와 함께 설명하고, 창작자의 개성을 살릴 수 있는 다양한 접근법을 제안하세요. 한국어의 아름다움을 살린 풍부한 표현으로 답변해주세요."""
            st.session_state.user_temperature = 0.8
            st.session_state.user_top_p = 0.95
            st.rerun()
    
    # Role Prompt
    st.session_state.user_role_prompt = st.text_area(
        "Role Prompt:",
        value=st.session_state.user_role_prompt,
        height=80
    )
    
    # System Prompt  
    st.session_state.user_system_prompt = st.text_area(
        "System Prompt:",
        value=st.session_state.user_system_prompt,
        height=100
    )
    
    # Temperature
    st.session_state.user_temperature = st.slider(
        "Temperature (창의성)",
        0.0, 1.0, 
        st.session_state.user_temperature,
        0.1
    )
    
    # Top-p
    st.session_state.user_top_p = st.slider(
        "Top-p (다양성)",
        0.0, 1.0,
        st.session_state.user_top_p, 
        0.05
    )
    
    # 현재 설정 표시
    with st.expander("📋 현재 설정", expanded=False):
        combined_prompt = f"{st.session_state.user_role_prompt}\n\n{st.session_state.user_system_prompt}"
        st.code(combined_prompt)
        st.write(f"Temperature: {st.session_state.user_temperature}")
        st.write(f"Top-p: {st.session_state.user_top_p}")
    
    # 세션 정보 표시
    st.subheader("📊 세션 정보")
    session_info = get_streamlit_session_info()
    st.code(f"대화 ID: {session_info.get('conversation_id', 'N/A')}")
    st.code(f"메시지 번호: {session_info.get('message_index', 'N/A')}")
    
    # 🐛 디버깅: 모니터링 상태 확인
    st.subheader("🔍 모니터링 상태")
    try:
        client = get_bedrock_client()
        is_monitored = hasattr(client.invoke_model, '_nr_monitored')
        is_auto_patched = hasattr(client.invoke_model, '__wrapped__')
        
        st.markdown(f"**모니터링 적용**: {'✅ 예' if is_monitored else '❌ 아니오'}")
        st.markdown(f"**자동 패치**: {'✅ 예' if is_auto_patched else '❌ 아니오'}")
        st.markdown(f"**앱 이름**: {APP_NAME}")
        
        # boto3 클라이언트 정보
        st.code(f"클라이언트 ID: {id(client)}")
        
    except Exception as e:
        st.error(f"모니터링 상태 확인 오류: {str(e)}")
    
    # 새 대화 시작
    if st.button("🔄 새 대화 시작"):
        st.session_state.messages = []
        st.session_state.last_response_data = {}
        # conversation_id는 자동으로 재생성됨
        st.rerun() 

with main_col:
    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    user_input = st.chat_input("질문을 입력하세요")
    
    if user_input:
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    start_time = time.time()
                    delay_times = {}
                    
                    # 🐌 LLM 호출 전 지연 적용
                    if st.session_state.delay_enabled and st.session_state.delay_before_llm:
                        delay_start = time.time()
                        apply_delay("before_llm")
                        delay_times["before_llm"] = time.time() - delay_start
                    
                    # 🚀 일반적인 boto3.client 호출 - 자동으로 모든 모니터링 처리!
                    bedrock_client = get_bedrock_client()
                    
                    # 첫 번째 메시지인 경우 모니터링 상태 한 번 더 확인
                    if len(st.session_state.messages) <= 1:
                        try:
                            if hasattr(bedrock_client.invoke_model, '__wrapped__'):
                                st.info("🎯 첫 번째 호출 - 모니터링 활성화 확인됨!")
                            else:
                                st.warning("⚠️ 첫 번째 호출 - 모니터링 미확인")
                        except Exception as check_error:
                            st.warning(f"모니터링 체크 오류: {str(check_error)}")
                    
                    # 시스템 프롬프트 구성
                    combined_system_prompt = f"{st.session_state.user_role_prompt}\n\n{st.session_state.user_system_prompt}"
                    
                    # Bedrock API 호출
                    llm_start_time = time.time()
                    response = bedrock_client.invoke_model(
                        modelId=MODEL_ID,
                        body=json.dumps({
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 1000,
                            "temperature": st.session_state.user_temperature,
                            "top_p": st.session_state.user_top_p,
                            "system": combined_system_prompt,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": user_input}]
                                }
                            ]
                        })
                    )
                    llm_duration = time.time() - llm_start_time
                    
                    # 🐌 LLM 응답 후 지연 적용
                    if st.session_state.delay_enabled and st.session_state.delay_after_llm:
                        delay_start = time.time()
                        apply_delay("after_llm")
                        delay_times["after_llm"] = time.time() - delay_start
                    
                    # 응답 처리 (라이브러리가 자동으로 텍스트 추출 및 New Relic 전송)
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    
                    # 응답 텍스트 추출 (fallback)
                    assistant_response = ""
                    if 'content' in response_body:
                        content = response_body['content']
                        if isinstance(content, list) and len(content) > 0:
                            if isinstance(content[0], dict) and 'text' in content[0]:
                                assistant_response = content[0]['text']
                    
                    if assistant_response:
                        st.markdown(assistant_response)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        
                        # 응답 데이터 저장 (평가 UI용) + 지연 정보 포함
                        usage = response_body.get("usage", {})
                        total_duration = int((time.time() - start_time) * 1000)
                        llm_duration_ms = int(llm_duration * 1000)
                        
                        delay_info = {}
                        total_delay_ms = 0
                        for stage, delay_time in delay_times.items():
                            delay_ms = int(delay_time * 1000)
                            delay_info[f"{stage}_delay_ms"] = delay_ms
                            total_delay_ms += delay_ms
                        
                        st.session_state.last_response_data = {
                            "response_time_ms": total_duration,
                            "llm_only_time_ms": llm_duration_ms,
                            "total_delay_ms": total_delay_ms,
                            "total_tokens": usage.get("total_token_count", 0) or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
                            "prompt_tokens": usage.get("input_token_count", 0) or usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_token_count", 0) or usage.get("output_tokens", 0),
                            "temperature": st.session_state.user_temperature,
                            "top_p": st.session_state.user_top_p,
                            **delay_info
                        }
                        
                        # 🐌 지연 정보 표시
                        if st.session_state.delay_enabled and delay_times:
                            with st.expander("🐌 지연 정보", expanded=False):
                                st.markdown(f"**총 처리 시간**: {total_duration}ms")
                                st.markdown(f"**실제 LLM 시간**: {llm_duration_ms}ms")
                                st.markdown(f"**총 지연 시간**: {total_delay_ms}ms")
                                
                                for stage, delay_time in delay_times.items():
                                    delay_ms = int(delay_time * 1000)
                                    stage_name = "LLM 호출 전" if stage == "before_llm" else "LLM 응답 후"
                                    st.markdown(f"- {stage_name}: {delay_ms}ms")
                                
                                efficiency = (llm_duration_ms / total_duration) * 100
                                st.markdown(f"**효율성**: {efficiency:.1f}% (실제 LLM 작업 시간 비율)")
                        
                    else:
                        st.error("응답을 추출할 수 없습니다.")
                        
                except Exception as e:
                    st.error(f"오류: {str(e)}")

# 📊 모니터링 정보 섹션
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 자동 수집되는 데이터")
    st.markdown("""
    **자동으로 New Relic에 전송:**
    - ✅ LlmCompletion (요청/응답, 토큰, 파라미터)
    - ✅ LlmUserRole (사용자 입력 이벤트)  
    - ✅ LlmSystemRole (시스템 프롬프트 이벤트)
    - ✅ LlmBedrockResponse (응답 상세 정보)
    - ✅ trace_id, completion_id (자동 생성)
    - ✅ conversation_id (세션 연동)
    
    **🐌 지연 테스트 추가 정보:**
    - ⏱️ 실제 LLM 응답 시간
    - 🐌 의도적 지연 시간
    - 📊 전체 처리 시간 및 효율성
    """)

with col2:
    # New Relic 쿼리 예제 (라이브러리 함수 사용)
    create_streamlit_nrql_queries(
        application_name=APP_NAME,
        conversation_id=session_info.get('conversation_id')
    )

# 📝 평가 UI (마지막 응답이 있을 때만 표시)
if (st.session_state.messages and 
    st.session_state.messages[-1]["role"] == "assistant" and
    st.session_state.last_response_data):
    
    st.markdown("---")
    
    # 라이브러리의 자동 평가 UI 사용 (New Relic 전송 포함)
    create_streamlit_evaluation_ui(
        # trace_id와 completion_id는 라이브러리가 자동으로 관리
        model_id=MODEL_ID,
        response_time_ms=st.session_state.last_response_data.get("response_time_ms"),
        total_tokens=st.session_state.last_response_data.get("total_tokens"),
        prompt_tokens=st.session_state.last_response_data.get("prompt_tokens"),
        completion_tokens=st.session_state.last_response_data.get("completion_tokens"),
        temperature=st.session_state.last_response_data.get("temperature"),
        top_p=st.session_state.last_response_data.get("top_p"),
        application_name=APP_NAME
    )

# 🐌 지연 테스트 정보 섹션
if st.session_state.delay_enabled:
    st.markdown("---")
    st.markdown("### 🐌 지연 테스트 정보")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delay_name, _ = DELAY_FUNCTIONS[st.session_state.delay_type]
        st.metric("지연 방법", delay_name)
    
    with col2:
        st.metric("지연 시간", f"{st.session_state.delay_duration}초")
    
    with col3:
        stages = []
        if st.session_state.delay_before_llm:
            stages.append("호출 전")
        if st.session_state.delay_after_llm:
            stages.append("응답 후")
        stage_text = ", ".join(stages) if stages else "없음"
        st.metric("적용 시점", stage_text)
    
    # 지연 방법별 설명
    with st.expander("🔍 지연 방법 설명", expanded=False):
        st.markdown("""
        **⏰ 단순 Sleep**: `time.sleep()`을 사용한 기본적인 대기
        
        **🔥 CPU 집약적**: SHA256 해시 계산을 통한 CPU 부하 생성
        
        **💾 메모리 집약적**: 대용량 배열 생성/삭제를 통한 메모리 부하
        
        **📁 I/O 시뮬레이션**: 임시 파일 생성/읽기/쓰기 작업
        
        **🌐 네트워크 시뮬레이션**: ThreadPoolExecutor를 사용한 동시 네트워크 호출 시뮬레이션
        
        **🔀 혼합 방법**: 위의 모든 방법을 순차적으로 실행
        
        이러한 다양한 지연 방법을 통해 실제 시스템에서 발생할 수 있는 다양한 성능 병목 상황을 시뮬레이션할 수 있습니다.
        """)

# 🧪 지연 테스트 및 벤치마크 섹션
if st.session_state.delay_enabled:
    st.markdown("---")
    st.markdown("### 🧪 지연 테스트 도구")
    
    # 지연 함수 벤치마크
    with st.expander("⚡ 지연 함수 벤치마크", expanded=False):
        st.markdown("각 지연 방법의 실제 성능을 테스트해볼 수 있습니다.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_duration = st.slider(
                "테스트 시간 (초)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                key="benchmark_duration"
            )
        
        with col2:
            test_method = st.selectbox(
                "테스트할 방법",
                options=list(DELAY_FUNCTIONS.keys()),
                format_func=lambda x: DELAY_FUNCTIONS[x][0],
                key="benchmark_method"
            )
        
        with col3:
            if st.button("🏃‍♂️ 벤치마크 실행", key="run_benchmark"):
                _, delay_func = DELAY_FUNCTIONS[test_method]
                
                with st.spinner(f"🧪 {DELAY_FUNCTIONS[test_method][0]} 테스트 중..."):
                    benchmark_start = time.time()
                    
                    try:
                        delay_func(test_duration)
                        actual_time = time.time() - benchmark_start
                        
                        st.success(f"✅ 테스트 완료!")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("예상 시간", f"{test_duration:.2f}초")
                        with col_b:
                            st.metric("실제 시간", f"{actual_time:.2f}초")
                        with col_c:
                            accuracy = (test_duration / actual_time) * 100
                            st.metric("정확도", f"{accuracy:.1f}%")
                        
                        if accuracy < 90:
                            st.warning(f"⚠️ 정확도가 낮습니다. 시스템 부하나 기타 요인이 영향을 주었을 수 있습니다.")
                        elif accuracy > 98:
                            st.success(f"🎯 매우 정확한 지연이 달성되었습니다!")
                        
                    except Exception as e:
                        actual_time = time.time() - benchmark_start
                        st.error(f"❌ 테스트 중 오류: {str(e)}")
                        st.info(f"경과 시간: {actual_time:.2f}초")

# 📈 성능 분석 및 모니터링 가이드
st.markdown("---")
st.markdown("### 📈 성능 분석 가이드")

with st.expander("🎯 이 도구의 활용 방법", expanded=False):
    st.markdown("""
    ### 🎯 late_levitwo.py 활용 가이드
    
    이 도구는 다음과 같은 목적으로 사용할 수 있습니다:
    
    #### 1. 📊 New Relic 모니터링 검증
    - 의도적인 지연을 통해 성능 이슈를 시뮬레이션
    - APM 대시보드에서 응답 시간 증가를 확인
    - 알림 및 임계값 설정의 정확성 검증
    
    #### 2. 🧪 성능 테스트
    - 다양한 부하 조건에서의 시스템 동작 확인
    - CPU, 메모리, I/O 병목 상황 시뮬레이션
    - 응답 시간 변화에 따른 사용자 경험 평가
    
    #### 3. 🔧 시스템 튜닝
    - 타임아웃 설정의 적절성 확인
    - 캐싱 전략의 효과 측정
    - 리소스 할당 최적화
    
    #### 4. 📚 교육 및 데모
    - 성능 모니터링의 중요성 설명
    - 다양한 지연 패턴의 영향 시연
    - 실시간 모니터링 도구 사용법 교육
    
    #### 추천 테스트 시나리오:
    
    **시나리오 1: 간단한 응답 시간 테스트**
    - 지연 방법: 단순 Sleep
    - 지연 시간: 1-3초
    - 적용 시점: LLM 호출 전
    
    **시나리오 2: CPU 부하 테스트**
    - 지연 방법: CPU 집약적
    - 지연 시간: 2-5초  
    - 적용 시점: LLM 호출 전 + 응답 후
    
    **시나리오 3: 복합 부하 테스트**
    - 지연 방법: 혼합 방법
    - 지연 시간: 3-7초
    - 적용 시점: LLM 호출 전
    
    각 시나리오 후 New Relic 대시보드에서 다음을 확인하세요:
    - 응답 시간 지표의 변화
    - 처리량(Throughput) 변화
    - 에러율 변화
    - 인프라 리소스 사용률
    """)

with st.expander("🔗 관련 New Relic 쿼리", expanded=False):
    st.markdown("""
    ### New Relic에서 지연 테스트 결과를 분석하기 위한 NRQL 쿼리:
    
    **1. 응답 시간 분포 확인:**
    ```nrql
    FROM Transaction SELECT histogram(duration) 
    WHERE appName = 'gen-ai-bedrock-late' 
    SINCE 1 hour ago
    ```
    
    **2. 지연 구간별 요청 수:**
    ```nrql
    FROM Transaction SELECT count(*) 
    WHERE appName = 'gen-ai-bedrock-late' 
    FACET buckets(duration, 1, 10, 5)
    SINCE 1 hour ago
    ```
    
    **3. LLM 호출 성능 분석:**
    ```nrql
    FROM LlmCompletion SELECT 
    average(duration), max(duration), min(duration), percentile(duration, 50, 95, 99)
    WHERE appName = 'gen-ai-bedrock-late'
    SINCE 1 hour ago
    ```
    
    **4. 시간대별 성능 변화:**
    ```nrql
    FROM Transaction SELECT average(duration) 
    WHERE appName = 'gen-ai-bedrock-late'
    TIMESERIES 5 minutes SINCE 1 hour ago
    ```
    
    **5. 에러율 모니터링:**
    ```nrql
    FROM Transaction SELECT percentage(count(*), WHERE error IS true)
    WHERE appName = 'gen-ai-bedrock-late'
    SINCE 1 hour ago
    ```
    """)

# 푸터
st.markdown("---")
st.markdown("🐌 nr-bedrock-observability-python v2.4.1 - 자동 패치 모드 + 지연 시뮬레이션")
st.caption("이 도구는 성능 테스트 및 모니터링 검증 목적으로 제작되었습니다. 프로덕션 환경에서는 지연 기능을 비활성화하세요.") 