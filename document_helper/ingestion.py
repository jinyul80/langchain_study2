import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def ingest_docs():
    doc_path = os.path.join(
        os.getcwd(),
        "document_helper",
        "langchain-docs",
        "api.python.langchain.com",
        "en",
        "latest",
    )
    db_path = os.path.join(os.getcwd(), "document_helper", "vector_db")

    loader = ReadTheDocsLoader(doc_path, encoding="utf-8")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace(
            os.path.join(os.getcwd(), "document_helper", "langchain-docs"), "https://"
        )
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} documents to Chroma...")

    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    Chroma.from_documents(
        documents,
        ollama_embeddings,
        collection_name="langchain-doc-index",
        persist_directory=db_path,
    )

    print("done!")


if __name__ == "__main__":
    print("\nIngesting...")

    ingest_docs()
