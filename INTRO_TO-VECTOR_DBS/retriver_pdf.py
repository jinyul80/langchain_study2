import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


load_dotenv()

if __name__ == "__main__":
    print("\nIngesting...")

    # PDF 문서 불러오기

    doc_path = os.path.join(os.getcwd(), "INTRO_TO-VECTOR_DBS", "2210.03629v3.pdf")
    db_path = os.path.join(os.getcwd(), "INTRO_TO-VECTOR_DBS", "data")
    lodder = PyPDFLoader(doc_path)
    document = lodder.load()

    # 문서를 FAISS 인덱스로 저장하기
    print("splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(document)
    print(f"created {len(docs)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(db_path, "faiss_index_react")

    # vectorstore 불러오기
    new_vectorstore = FAISS.load_local(
        db_path, embeddings, "faiss_index_react", allow_dangerous_deserialization=True
    )
    print("Vector store loaded successfully!")

    # Retrieval QA Chat Prompt
    llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

    query = """
        Give me the gist of ReAct in 3 sentences.
        
        모든 대답은 한글로 작성하고, 출처를 표시해 주세요.
        """

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    print("\nAnswer: ")
    print(result["answer"])
