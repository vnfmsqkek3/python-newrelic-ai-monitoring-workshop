"""
AWS Bedrock 모니터링 애플리케이션 (Streamlit) - Knowledge Base 없이 직접 Bedrock 호출

이 앱은 AWS Bedrock API 호출을 모니터링하고 NewRelic에 데이터를 전송합니다.
지식 기반 없이 Claude 3.5 Sonnet 모델을 직접 사용하여 질문에 답변하고 사용자 피드백을 수집합니다.
"""

import streamlit as st
import boto3
import json
import uuid
import time
import os
import newrelic.agent
from nr_bedrock_observability import (
    monitor_bedrock, 
    # 평가 수집 도구
    init_response_evaluation_collector,
    ensure_evaluation_state,
    update_evaluation_state,
    create_update_callback,
    create_evaluation_ui,
    create_evaluation_debug_ui,
    send_evaluation_with_newrelic_agent,
    get_evaluation_collector,
    reset_evaluation_collector,
    # 새로 추가된 대시보드 헬퍼 함수
    record_role_based_events,
    # record_search_results,  # Knowledge Base 없이는 사용하지 않음
    record_bedrock_response,
    extract_claude_response_text,
    get_sample_nrql_queries,
    # search_knowledge_base  # Knowledge Base 없이는 사용하지 않음
)

# NewRelic 라이센스 키 설정 - 실제 환경에서는 환경 변수나 보안 방식으로 관리해야 합니다
os.environ["NEW_RELIC_LICENSE_KEY"] = "XXXXXXXXXXXX"  # 실제 라이센스 키로 변경 필요

# 상수 정의
REGION = "ap-northeast-2"
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
# KNOWLEDGE_BASE_ID = "VNJYWIISJU"  # Knowledge Base 없이는 사용하지 않음
# KNOWLEDGE_BASE_NAME = "knowledge-base-quick-start-ycffx"  # Knowledge Base 없이는 사용하지 않음
APP_NAME = "gen-ai-bedrock-app"

# 페이지 설정
st.set_page_config(
    page_title="Bedrock 모니터링 앱 (직접 호출)",
    page_icon="🤖",
    layout="wide"
)

# 뉴렐릭 설정 확인
try:
    nr_app = newrelic.agent.application()
    if nr_app:
        st.success(f"뉴렐릭 애플리케이션이 성공적으로 초기화되었습니다: {nr_app.name}")
    else:
        st.warning(f"뉴렐릭 애플리케이션을 찾을 수 없습니다. 라이센스 키를 확인하세요: {os.environ.get('NEW_RELIC_LICENSE_KEY', '').replace('FFFFNRAL', '****')}")
except Exception as e:
    st.error(f"뉴렐릭 초기화 오류: {str(e)}")

# 세션 상태 초기화
if "trace_id" not in st.session_state:
    st.session_state.trace_id = str(uuid.uuid4())
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_completion_id" not in st.session_state:
    st.session_state.current_completion_id = None
if "raw_result" not in st.session_state:
    st.session_state.raw_result = {}
# Knowledge Base 관련 세션 상태는 사용하지 않음
# if "current_search_results" not in st.session_state:
#     st.session_state.current_search_results = []
if "current_system_prompt" not in st.session_state:
    st.session_state.current_system_prompt = ""
if "message_count" not in st.session_state:
    st.session_state.message_count = 0
# 사용자 설정 가능한 파라미터들 초기화
if "user_role_prompt" not in st.session_state:
    st.session_state.user_role_prompt = "당신은 도움이 되는 AI 어시스턴트입니다."
if "user_system_prompt" not in st.session_state:
    st.session_state.user_system_prompt = "사용자의 질문에 명확하고 정확하게 답변해주세요. 한국어로 답변해주세요."
if "user_temperature" not in st.session_state:
    st.session_state.user_temperature = 0.3
if "user_top_p" not in st.session_state:
    st.session_state.user_top_p = 0.9

# 평가 수집기 초기화
if "response_evaluation_collector" not in st.session_state:
    try:
        # 새로운 평가 수집기 초기화
        init_response_evaluation_collector(
            application_name=APP_NAME,
            trace_id=st.session_state.trace_id,
            completion_id=None,
            session_id=st.session_state.conversation_id,
            collector_session_key="response_evaluation_collector"
        )
    except Exception as e:
        st.warning(f"평가 수집기 초기화 중 오류: {str(e)}")

# Bedrock 클라이언트 설정
@st.cache_resource
def get_bedrock_client():
    """Bedrock 런타임 클라이언트를 생성하고 모니터링 설정"""
    bedrock_client = boto3.client('bedrock-runtime', region_name=REGION)
    monitored_client = monitor_bedrock(bedrock_client, {
        'application_name': APP_NAME,
        'new_relic_api_key': os.environ.get("NEW_RELIC_LICENSE_KEY")
    })
    return monitored_client

# Knowledge Base 없이 직접 Bedrock 호출하는 워크플로우
def run_direct_bedrock_workflow(user_query):
    """직접 Bedrock 워크플로우 실행: Knowledge Base 없이 Claude 3.5 Sonnet에 직접 질문"""
    start_time = time.time()
    
    # 새 완성 ID 생성하고 기존 대화 ID 유지
    trace_id = str(uuid.uuid4())
    completion_id = str(uuid.uuid4())
    conversation_id = st.session_state.conversation_id
    st.session_state.current_completion_id = completion_id
    st.session_state.message_count += 1
    message_index = st.session_state.message_count
    
    # 트레이스 ID 디버깅 출력
    st.info(f"직접 호출 - 대화 ID: {conversation_id}, 트레이스 ID: {trace_id}, 완성 ID: {completion_id}, 메시지 순서: {message_index}")
    
    # 응답 평가 수집기 초기화 또는 업데이트
    try:
        init_response_evaluation_collector(
            application_name=APP_NAME,
            trace_id=trace_id,
            completion_id=completion_id,
            user_id=None,
            session_id=conversation_id
        )
    except Exception as e:
        st.warning(f"응답 평가 수집기 초기화 중 오류: {str(e)}")
    
    # Knowledge Base 검색 단계는 생략
    # 시스템 프롬프트와 사용자 쿼리 구성
    # role prompt와 system prompt를 합쳐서 전체 system prompt 생성
    combined_system_prompt = f"{st.session_state.user_role_prompt}\n\n{st.session_state.user_system_prompt}"
    system_prompt = combined_system_prompt
    
    # 시스템 프롬프트 저장
    st.session_state.current_system_prompt = system_prompt
    
    # 사용자 쿼리를 직접 사용 (Knowledge Base 컨텍스트 없이)
    user_content = user_query
    
    # 역할별 이벤트 기록 - Knowledge Base 검색 결과 없이
    record_role_based_events(
        user_query=user_query,
        system_prompt=system_prompt,
        search_results=[],  # 검색 결과 없음
        context_text="",    # 컨텍스트 없음
        trace_id=trace_id,
        completion_id=completion_id,
        application_name=APP_NAME,
        conversation_id=conversation_id,
        message_index=message_index
    )
    
    # Knowledge Base 검색 결과 기록은 생략
    # record_search_results(...) 호출 없음
    
    # Bedrock 요청 구성 - Claude 3.5 Sonnet 형식에 맞춤
    request = {
        'modelId': MODEL_ID,
        'body': json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            'max_tokens': 1000,
            'temperature': st.session_state.user_temperature,
            'top_p': st.session_state.user_top_p,
            'system': system_prompt,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': user_content
                        }
                    ]
                }
            ]
        })
    }
    
    # 모델 파라미터 추출 (temperature와 top_p)
    request_body = json.loads(request['body'])
    temperature = request_body.get('temperature', 0.3)
    top_p = request_body.get('top_p', 0.9)
    
    # Bedrock 요청 실행
    try:
        # Bedrock 호출
        bedrock_client = get_bedrock_client()
        response = bedrock_client.invoke_model(**request)
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        # 응답 텍스트 추출 - 라이브러리 함수 사용
        assistant_response = extract_claude_response_text(response_body)
        
        # Bedrock 응답을 New Relic에 기록 - Knowledge Base 정보 없이
        record_bedrock_response(
            assistant_response=assistant_response,
            response_body=response_body,
            trace_id=trace_id,
            completion_id=completion_id,
            application_name=APP_NAME,
            model_id=MODEL_ID,
            kb_id=None,  # Knowledge Base 없음
            kb_name=None,  # Knowledge Base 없음
            conversation_id=conversation_id,
            message_index=message_index,
            response_time_ms=int((time.time() - start_time) * 1000),
            temperature=temperature,
            top_p=top_p
        )
        
        # 토큰 사용량 추출 - Claude 3.5 호환
        usage = response_body.get("usage", {})
        # 일반 Bedrock 토큰 필드 확인
        total_tokens = usage.get("total_token_count", 0)
        input_tokens = usage.get("input_token_count", 0)
        output_tokens = usage.get("output_token_count", 0)
        
        # Claude 3.5 토큰 필드 확인
        if total_tokens == 0:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
        
        # 전체 처리 완료 시간 측정
        total_duration = int((time.time() - start_time) * 1000)  # 밀리초로 변환
        
        return assistant_response, {
            # "search_results": [],  # 검색 결과 없음
            # "search_time_ms": 0,   # 검색 시간 없음
            "llm_time_ms": total_duration,
            "total_time_ms": total_duration,
            "trace_id": trace_id,
            "completion_id": completion_id,
            "token_count": output_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens": input_tokens,
            "model_id": MODEL_ID,
            "kb_id": None,  # Knowledge Base 없음
            "kb_name": None,  # Knowledge Base 없음
            "kb_used_in_query": False,  # Knowledge Base 사용 안함
            "response_time_ms": total_duration,
            "temperature": temperature,
            "top_p": top_p
        }
        
    except Exception as e:
        st.error(f"Bedrock 호출 오류: {str(e)}")
        return None, None

# UI 구성 함수
def build_ui():
    """Streamlit UI 구성"""
    st.title("🤖 AWS Bedrock 직접 호출 Q&A + 모니터링")
    
    # 레이아웃: 메인 컬럼과 사이드바
    main_col, info_col = st.columns([2, 1])
    
    with info_col:
        st.header("앱 정보")
        st.markdown(f"""
        **모델**: {MODEL_ID}
        
        **Knowledge Base**: 사용 안함 (직접 호출)
        
        **리전**: {REGION}
        
        **NewRelic 앱 이름**: {APP_NAME}
        """)
        
        # 모델 파라미터 설정 섹션 추가
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
        
        # 기본 설정 복원 버튼
        if st.button("🔄 기본 설정 복원", help="모든 설정을 기본값으로 되돌립니다"):
            st.session_state.user_role_prompt = "당신은 도움이 되는 AI 어시스턴트입니다."
            st.session_state.user_system_prompt = "사용자의 질문에 명확하고 정확하게 답변해주세요. 한국어로 답변해주세요."
            st.session_state.user_temperature = 0.3
            st.session_state.user_top_p = 0.9
            st.rerun()
        
        st.markdown("---")  # 구분선 추가
        
        # Role Prompt 입력
        st.subheader("Role Prompt")
        new_role_prompt = st.text_area(
            "모델의 역할을 정의하세요:",
            value=st.session_state.user_role_prompt,
            height=80,
            key="role_prompt_input",
            help="예: '당신은 의료 전문가입니다', '당신은 코딩 튜터입니다' 등"
        )
        if new_role_prompt != st.session_state.user_role_prompt:
            st.session_state.user_role_prompt = new_role_prompt
        
        # System Prompt 입력
        st.subheader("System Prompt")
        new_system_prompt = st.text_area(
            "구체적인 행동 지침을 입력하세요:",
            value=st.session_state.user_system_prompt,
            height=100,
            key="system_prompt_input",
            help="모델이 어떻게 행동해야 하는지 구체적으로 명시하세요"
        )
        if new_system_prompt != st.session_state.user_system_prompt:
            st.session_state.user_system_prompt = new_system_prompt
        
        # Temperature 슬라이더
        st.subheader("Temperature")
        new_temperature = st.slider(
            "창의성 수준 (낮을수록 일관성 높음)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.user_temperature,
            step=0.1,
            key="temperature_slider",
            help="0.0: 매우 일관적, 1.0: 매우 창의적"
        )
        if new_temperature != st.session_state.user_temperature:
            st.session_state.user_temperature = new_temperature
        
        # Top-p 슬라이더
        st.subheader("Top-p (Nucleus Sampling)")
        new_top_p = st.slider(
            "응답 다양성 제어",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.user_top_p,
            step=0.1,
            key="top_p_slider",
            help="0.0: 가장 확률 높은 단어만, 1.0: 모든 단어 고려"
        )
        if new_top_p != st.session_state.user_top_p:
            st.session_state.user_top_p = new_top_p
            
        # 현재 설정 미리보기
        with st.expander("현재 파라미터 설정", expanded=False):
            st.markdown("**최종 System Prompt:**")
            combined_preview = f"{st.session_state.user_role_prompt}\n\n{st.session_state.user_system_prompt}"
            st.code(combined_preview)
            st.markdown(f"**Temperature:** {st.session_state.user_temperature}")
            st.markdown(f"**Top-p:** {st.session_state.user_top_p}")
        
        # 트레이스 정보
        st.subheader("트레이스 정보")
        
        # ID 설명 추가
        with st.expander("ID 설명", expanded=True):
            st.markdown("""
            **대화 ID (Conversation ID)**: 사용자의 전체 대화 세션을 식별하는 고유 ID입니다. 새로운 대화 세션이 시작될 때 생성되며, 사용자가 여러 메시지를 주고받는 동안 동일하게 유지됩니다. 전체 대화 흐름을 추적할 때 사용됩니다.
            
            **트레이스 ID (Trace ID)**: 각 메시지 교환을 추적하기 위한 ID입니다. 새로운 사용자 입력이 있을 때마다 새로운 트레이스 ID가 생성됩니다. 단일 요청의 전체 처리 과정(Bedrock 직접 호출)을 추적하는 데 사용됩니다.
            
            **완성 ID (Completion ID)**: LLM 응답 완성에 대한 고유 ID입니다. Bedrock API 호출의 결과를 식별하고 추적하는 데 사용됩니다. 각 완성은 특정 사용자 질문에 대한 모델의 응답을 나타냅니다.
            """)
        
        st.code(f"대화 ID: {st.session_state.conversation_id}")
        st.code(f"트레이스 ID: {st.session_state.trace_id}")
        if st.session_state.current_completion_id:
            st.code(f"완성 ID: {st.session_state.current_completion_id}")
            
            # 모델 평가 분석 NRQL 예시 추가
            with st.expander("모델 평가 분석 쿼리", expanded=False):
                st.markdown("### 모델 평가 데이터 분석을 위한 NRQL 쿼리")
                
                # 라이브러리 함수를 사용하여 샘플 쿼리 가져오기 (Knowledge Base ID 없이)
                sample_queries = get_sample_nrql_queries(
                    trace_id=st.session_state.trace_id,
                    completion_id=st.session_state.current_completion_id,
                    conversation_id=st.session_state.conversation_id,
                    kb_id=None  # Knowledge Base 없음
                )
                
                # 쿼리 결과가 없는 이유에 대한 설명 추가
                st.warning("""
                **참고**: 쿼리 결과가 없는 경우 다음과 같은 이유가 있을 수 있습니다:
                
                1. 아직 평가 데이터가 수집되지 않았습니다. 먼저 몇 개의 모델 평가를 제출해보세요.
                2. 이벤트 타입 이름이 실제 New Relic에 기록된 이름과 다를 수 있습니다.
                3. New Relic 계정 설정에서 사용자 정의 이벤트 수집이 활성화되어 있는지 확인하세요.
                4. 시간 범위를 더 넓게 설정하여 확인해 보세요.
                
                평가 제출 후 데이터가 New Relic에 표시되는 데 약간의 시간이 걸릴 수 있습니다.
                """)
                
                # 샘플 쿼리 표시
                for title, query in sample_queries.items():
                    st.markdown(f"**{title}:**")
                    st.code(query)
        
        # 시스템 프롬프트 표시
        if st.session_state.current_system_prompt:
            with st.expander("시스템 프롬프트", expanded=False):
                st.code(st.session_state.current_system_prompt)
        
        # Knowledge Base 검색 결과는 표시하지 않음 (검색 안함)
        # if st.session_state.current_search_results:
        #     with st.expander("검색 결과", expanded=False):
        #         ...
        
        # 실행 정보 표시
        if st.session_state.raw_result:
            with st.expander("실행 정보", expanded=False):
                # st.metric("검색 시간", f"{st.session_state.raw_result.get('search_time_ms', 0)} ms")
                st.metric("LLM 응답 시간", f"{st.session_state.raw_result.get('llm_time_ms', 0)} ms")
                st.metric("총 처리 시간", f"{st.session_state.raw_result.get('total_time_ms', 0)} ms")
                st.metric("토큰 수", st.session_state.raw_result.get('token_count', 0))
                
                # 사용된 파라미터 정보 추가
                st.markdown("**사용된 파라미터:**")
                st.markdown(f"- Temperature: {st.session_state.raw_result.get('temperature', 'N/A')}")
                st.markdown(f"- Top-p: {st.session_state.raw_result.get('top_p', 'N/A')}")
                st.markdown(f"- 총 토큰: {st.session_state.raw_result.get('total_tokens', 'N/A')}")
                st.markdown(f"- 프롬프트 토큰: {st.session_state.raw_result.get('prompt_tokens', 'N/A')}")
        
        # 새 대화 시작 버튼
        if st.button("새 대화 시작"):
            # 이전 평가 상태 키를 모두 찾아서 초기화
            eval_keys = [key for key in st.session_state.keys() if key.startswith("eval_")]
            for key in eval_keys:
                del st.session_state[key]
                
            # 이전 슬라이더/위젯 키 초기화
            widget_keys = [key for key in st.session_state.keys() if 
                          any(key.startswith(prefix) for prefix in 
                              ["overall_score_", "relevance_score_", "accuracy_score_", 
                               "completeness_score_", "coherence_score_", "helpfulness_score_",
                               "response_time_score_", "query_type_", "domain_", "feedback_comment_",
                               "submit_", "reset_log_", "test_eval_"])]
            for key in widget_keys:
                if key in st.session_state:
                    del st.session_state[key]
            
            # 기본 세션 상태 초기화
            st.session_state.trace_id = str(uuid.uuid4())
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.current_completion_id = None
            st.session_state.raw_result = {}
            # st.session_state.current_search_results = []  # Knowledge Base 없으니 사용 안함
            # 현재 시스템 프롬프트를 사용자 설정으로 업데이트
            st.session_state.current_system_prompt = f"{st.session_state.user_role_prompt}\n\n{st.session_state.user_system_prompt}"
            st.session_state.message_count = 0
            
            # 응답 평가 수집기 초기화
            try:
                # 응답 평가 수집기 세션 상태 초기화
                if "response_evaluation_collector" in st.session_state:
                    # 라이브러리 함수 사용하여 평가 수집기 초기화
                    reset_evaluation_collector(collector_session_key="response_evaluation_collector")
                
                # 새로운 수집기 초기화
                init_response_evaluation_collector(
                    application_name=APP_NAME,
                    trace_id=st.session_state.trace_id,
                    completion_id=None,
                    session_id=st.session_state.conversation_id,
                    collector_session_key="response_evaluation_collector"
                )
            except Exception as e:
                st.warning(f"평가 수집기 초기화 중 오류: {str(e)}")
                
            st.rerun()
    
    with main_col:
        # 채팅 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 입력
        user_input = st.chat_input("질문을 입력하세요")
        
        if user_input:
            # 사용자 메시지 표시 (원본 질문만 표시)
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # 메시지 저장 (원본 질문만 저장)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 응답 생성 - Knowledge Base 없이 직접 Bedrock 호출
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    response, result_info = run_direct_bedrock_workflow(user_input)
                    
                    if response:
                        st.markdown(response)
                        
                        # 결과 정보 저장
                        st.session_state.raw_result = result_info
                        
                        # 메시지 저장
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # 응답 평가 수집기 업데이트
                        try:
                            if result_info:
                                # 수집기 업데이트
                                init_response_evaluation_collector(
                                    application_name=APP_NAME,
                                    trace_id=result_info.get('trace_id'),
                                    completion_id=result_info.get('completion_id'),
                                    session_id=st.session_state.conversation_id,
                                    collector_session_key="response_evaluation_collector"
                                )
                        except Exception as e:
                            st.warning(f"응답 평가 수집기 업데이트 중 오류: {str(e)}")
                            
                    else: # if response (no response generated)
                        st.warning("모델로부터 응답을 받지 못했습니다.")
                # 여기서 with st.spinner 끝
            # 여기서 with st.chat_message("assistant") 끝
        # 여기서 if user_input 끝

    # --- 모델 응답 평가 UI 섹션 ---
    # 가장 마지막 메시지가 어시스턴트의 응답일 경우에만 평가 UI를 표시합니다.
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        
        eval_key = f"eval_{st.session_state.message_count}"
        
        # 현재 raw_result 정보를 안전하게 가져오기
        current_raw_result = st.session_state.get("raw_result", {})
        
        # 평가 섹션 표시
        st.markdown("### 모델 응답 평가")
        
        # 라이브러리 함수를 사용한 Streamlit 응답 평가 UI - Knowledge Base 정보 없이
        try:
            create_evaluation_ui(
                eval_key=eval_key,
                trace_id=current_raw_result.get('trace_id'),
                completion_id=current_raw_result.get('completion_id'),
                model_id=MODEL_ID,
                kb_id=None,  # Knowledge Base 없음
                kb_name=None,  # Knowledge Base 없음
                kb_used_in_query=False,  # Knowledge Base 사용 안함
                response_time_ms=current_raw_result.get('total_time_ms'),
                total_tokens=current_raw_result.get('total_tokens'),
                prompt_tokens=current_raw_result.get('prompt_tokens'),
                completion_tokens=current_raw_result.get('token_count'),
                temperature=current_raw_result.get('temperature'),
                top_p=current_raw_result.get('top_p'),
                application_name=APP_NAME,
                use_number_input=True,  # 슬라이더 대신 숫자 입력 사용
                submit_button_text="평가 제출",
                evaluation_source="streamlit"
            )
        except Exception as e:
            st.error(f"평가 UI 생성 중 오류 발생: {str(e)}")
            # 기본 평가 UI 대신 수동 평가 제출 폼 표시
            st.write("대체 평가 방식을 사용합니다:")
            with st.form("manual_evaluation_form"):
                overall_score = st.number_input("전체 만족도", min_value=1, max_value=10, value=5)
                relevance = st.number_input("질문 관련성", min_value=1, max_value=10, value=5)
                accuracy = st.number_input("정확성", min_value=1, max_value=10, value=5)
                submit = st.form_submit_button("평가 제출")
                
                if submit:
                    try:
                        # 수동으로 평가 제출 - 라이브러리 함수 직접 사용 (Knowledge Base 정보 없이)
                        evaluation_data = {
                            "model_id": MODEL_ID,
                            "overall_score": overall_score,
                            "relevance_score": relevance,
                            "accuracy_score": accuracy,
                            "kb_id": None,  # Knowledge Base 없음
                            "kb_name": None,  # Knowledge Base 없음
                            "evaluation_source": "streamlit-manual",
                            "trace_id": current_raw_result.get('trace_id'),
                            "completion_id": current_raw_result.get('completion_id'),
                            "temperature": current_raw_result.get('temperature'),
                            "top_p": current_raw_result.get('top_p'),
                            "application_name": APP_NAME
                        }
                        
                        result = send_evaluation_with_newrelic_agent(
                            event_data=evaluation_data,
                            event_type="LlmUserResponseEvaluation"
                        )
                        
                        if result:
                            st.success("평가가 성공적으로 제출되었습니다!")
                    except Exception as submit_error:
                        st.error(f"평가 제출 중 오류: {str(submit_error)}")
        
        # 개발자 도구
        st.markdown("### 개발자 도구")
        show_debug = st.checkbox("디버깅 정보 표시", value=False, key=f"show_debug_{eval_key}")
        
        if show_debug:
            # 간단한 디버깅 정보 직접 표시
            st.code(f"""
            트레이스 ID: {current_raw_result.get('trace_id')}
            완성 ID: {current_raw_result.get('completion_id')}
            모델 ID: {MODEL_ID}
            응답 시간: {current_raw_result.get('total_time_ms')} ms
            총 토큰: {current_raw_result.get('total_tokens')}
            Knowledge Base 사용: 없음 (직접 호출)
            """)
            
            # 수동 테스트 평가 전송 버튼
            if st.button("테스트 평가 전송", key=f"test_eval_manual_{eval_key}"):
                try:
                    # 테스트 평가 제출 - 라이브러리 함수 직접 사용 (Knowledge Base 정보 없이)
                    evaluation_data = {
                        "model_id": MODEL_ID,
                        "overall_score": 8,
                        "kb_id": None,  # Knowledge Base 없음
                        "kb_name": None,  # Knowledge Base 없음
                        "evaluation_source": "streamlit-manual-test",
                        "trace_id": current_raw_result.get('trace_id'),
                        "completion_id": current_raw_result.get('completion_id'),
                        "temperature": current_raw_result.get('temperature'),
                        "top_p": current_raw_result.get('top_p'),
                        "application_name": APP_NAME
                    }
                    
                    test_result = send_evaluation_with_newrelic_agent(
                        event_data=evaluation_data,
                        event_type="LlmUserResponseEvaluation"
                    )
                    
                    if test_result:
                        st.success(f"테스트 평가 전송 성공: ID={test_result.get('id')}")
                except Exception as e:
                    st.error(f"테스트 평가 전송 오류: {str(e)}")

    # 푸터
    st.markdown("---")
    st.markdown(
        "이 앱은 nr-bedrock-observability-python v2.0.2 라이브러리를 사용하여 "
        "AWS Bedrock API 직접 호출을 모니터링하며, "
        "Knowledge Base 없이 Claude 3.5 Sonnet 모델과 직접 대화할 수 있습니다. "
        "temperature와 top_p 파라미터를 포함한 토큰 처리와 이벤트 수집이 개선된 Streamlit 평가 UI를 제공합니다."
    )

if __name__ == "__main__":
    build_ui() 