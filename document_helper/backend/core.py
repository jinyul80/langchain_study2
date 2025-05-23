import os
from typing import Any
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever


from langchain import hub

load_dotenv()

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: list[dict[str, Any]] = []):
    db_path = os.path.join(os.getcwd(), "document_helper", "vector_db")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=INDEX_NAME,
    )

    llm = ChatGroq(temperature=0, model="qwen-qwq-32b")
    # llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-2.0-flash")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    # Chat 이력을 포함한 검색기 생성
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retrieval = create_history_aware_retriever(
        llm=llm, retriever=vectorstore.as_retriever(), prompt=rephrase_prompt
    )

    retrival_chain = create_retrieval_chain(
        retriever=history_aware_retrieval, combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }

    return new_result


if __name__ == "__main__":
    print("Retriving...")

    query = """
        What is langchain chain?
        
        모든 대답은 한글로 작성하고, 출처를 표시해 주세요.
        """

    result = run_llm(query)

    print("\nAnswer: ")
    print(result["result"])
