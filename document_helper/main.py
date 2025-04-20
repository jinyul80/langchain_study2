from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Langchain Chatbot - Documentation Helper Bot")

# 질문 입력
prompt = st.text_input("Prompt", placeholder="Enter your question here...")

# 세션 상태 초기화
if (
    "user_prompt_history" not in st.session_state
    and "chat_answers_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answers_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: set[str]):
    """출처 목록을 문자열로 변환합니다.

    Args:
        source_urls (set[str]): 출처 목록

    Returns:
        str: 변환된 출처 문자열
    """

    if not source_urls:
        return ""

    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "출처:\n"
    for idx, source in enumerate(sources_list):
        sources_string += f"{idx + 1}. {source}\n"

    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        query = f"{prompt} \n모든 대답은 한글로 대답해주세요."
        generated_response = run_llm(
            query=query, chat_history=st.session_state["chat_history"]
        )

        sources = set(
            [
                doc.metadata.get("source")
                for doc in generated_response["source_documents"]
            ]
        )

        formatted_response = (
            f"{generated_response.get("result")} \n\n{create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(
            ("ai", generated_response.get("result"))
        )


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response, is_user=False)
