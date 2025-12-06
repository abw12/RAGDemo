from pathlib import Path

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "chroma_db"


def load_documents():
    docs = []
    # Load all CSV files in data/
    for csv_path in DATA_DIR.glob("*.csv"):
        # Each row becomes a document you can choose which column to use as text
        loader = CSVLoader(
            file_path=str(csv_path),
            encoding="utf-8",
            # If you want only some columns, you can use csv_args.
            # csv_args={"delimiter": ","}
        )
        docs.extend(loader.load())
    return docs

def split_documents(docs):
    # FOr csv rows are often small, we can use large chunk size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100 ## characters to overlap for maintianing the continuity 
    )

    return splitter.split_documents(docs)

def build_and_persist_vector_store(chunks):
    #Embedding via ollama (nomic-embed-text)
    ollama_embedding = OllamaEmbeddings(
        model = "nomic-embed-text" # using the model pulled in ollama
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=ollama_embedding,
        persist_directory=str(VECTOR_DIR),
    )

    vectordb.persist()
    return vectordb

def main():
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} raw docs (rows)")

    chunks = split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks")

    #Preview
    # for c in chunks[:3]:
    #     print("-"*80)
    #     print("METADATA:",c.metadata)
    #     print(c.page_content[:400])

    vectordb = build_and_persist_vector_store(chunks)
    print("Vector store built and persisted at:", VECTOR_DIR)

if __name__ == "__main__":
    main()