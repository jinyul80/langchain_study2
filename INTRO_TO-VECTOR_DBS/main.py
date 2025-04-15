import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain import hub

load_dotenv()


if __name__ == "__main__":
    print("Retriving...")

    db_path = os.path.join(os.getcwd(), "INTRO_TO-VECTOR_DBS", "data")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

    query = """
        what is Pinecone in machine learning?
        
        모든 대답은 한글로 작성하고, 출처를 표시해 주세요.
        """

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="mediumblog1",
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    print("\nAnswer: ")
    print(result["answer"])
