from pathlib import Path
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
from sympy import per

from index_documents import VECTOR_DIR

BASE_DIR = Path(__file__).parent.parent
VECTOR_DIR = BASE_DIR / "chroma_db"

def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR)
    )

    query="How many runs scored by AB de Villiers in ODI?"
    print(f"Question: {query}")

    docs_and_scores = vectordb.similarity_search_with_score(query,k=6)
    print("\nRetrived",len(docs_and_scores),"docs")

    for i, (d,score) in enumerate(docs_and_scores):
        print(f"\n--- Doc #{i+1} --- (score={score})")
        print("METADATA:", d.metadata)
        print("CONTENT:", d.page_content[:400].replace("\n", " "))
if __name__ == "__main__":
    main()