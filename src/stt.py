"""STT(음성 인식) 헬퍼. 기획서 5.2 참고.

Diarization(화자 분리)은 구현하지 않는다 - 현재 파이프라인의 태깅 로직
([src/nodes/minutes.py](nodes/minutes.py))은 "발언 시점이 아니라 내용이
어느 아이디어에 속하는가"로만 판단하므로 화자 식별이 결과에 영향을 주지
않고, pyannote 등 별도 모델 인증·정렬 로직을 추가하는 비용 대비 실익이
낮다고 판단해 스코프에서 뺐다.

OpenAI 오디오 트랜스크립션 API를 직접 호출한다(이미지 생성과 동일하게
채팅 모델이 아니라 별도 엔드포인트라 [src/llm.py](llm.py)의
build_chat_model 대상이 아니다 - [src/nodes/image.py](nodes/image.py)와
같은 패턴).
"""
import os

from openai import OpenAI

STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
# ISO-639-1 코드. 언어를 안 주면 자동 감지에 맡기는데, 회의가 전부 한국어라
# 굳이 감지에 맡길 이유가 없고, 자동 감지가 특히 짧은 발화에서 틀리면
# 엉뚱한 언어로 받아적을 수 있다 - 처음부터 한국어로 고정한다.
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "ko")


def transcribe(audio_file) -> str:
    """오디오 파일(파일 경로 또는 파일류 객체)을 텍스트로 변환한다.

    audio_file은 OpenAI SDK가 받는 file 파라미터 형식 그대로 전달한다 -
    Streamlit의 `st.audio_input()`이 반환하는 UploadedFile을 그대로 넘겨도
    된다(파일류 객체 + 파일명 속성을 가짐).
    """
    client = OpenAI()
    response = client.audio.transcriptions.create(
        model=STT_MODEL, file=audio_file, language=STT_LANGUAGE
    )
    return response.text
