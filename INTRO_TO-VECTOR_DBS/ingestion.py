import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

if __name__ == "__main__":
    print("\nIngesting...")

    doc_path = os.path.join(os.getcwd(), "INTRO_TO-VECTOR_DBS", "mediumblog1.txt")
    db_path = os.path.join(os.getcwd(), "INTRO_TO-VECTOR_DBS", "data")
    lodder = TextLoader(doc_path, encoding="utf-8")
    document = lodder.load()

    print("splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_documents(
        texts,
        ollama_embeddings,
        collection_name="mediumblog1",
        persist_directory=db_path,
    )

    print("done!")
