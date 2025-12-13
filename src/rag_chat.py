from pathlib import Path

from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from index_documents import VECTOR_DIR

BASE_DIR = Path(__file__).parent.parent
VECTOR_DIR = BASE_DIR / "chroma_db"

def load_vector_store():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR),
    )
    return vectordb

def build_rag_chain():
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.1
    )

    vectordb = load_vector_store()
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

    # RAG prompt template
    prompt = ChatPromptTemplate.from_messages([
       ("ai",
         "You are a helpful cricket statistics assistant. "
         "Answer the user's question ONLY using the provided context from the CSV data. "
         "If the answer is not in the context, say "
         "'I don't find this in my CSV knowledge base.'"),
        ("human",
         "Context:\n{context}\n\nQuestion: {question}")
    ])

    def rag_answer(question:str):
        # 1. Retrieve relevant docs from vector db
        docs = retriever.invoke(question) ## The question is turned into a vector when you call invoke function

        print("\n[DEBUG] Retrieved", len(docs), "documents")
        for i,d in enumerate(docs):
            print(f"\n[DEBUG] --- Doc #{i+1} ---")
            print("[DEBUG] METADATA:", d.metadata)
            print("[DEBUG] CONTENT PREVIEW:", d.page_content[:300].replace("\n", " "))
        
        if not docs:
            return "No documents were retrieved from the vector store. Please check indexing."
        
        context_text = "\n\n".join([d.page_content for d in docs])

        # Optional: see the actual context passed to LLM
        print("\n[DEBUG] ===== CONTEXT SENT TO LLM =====")
        print(context_text[:1000])   # print first 1000 chars only
        print("[DEBUG] ===== END CONTEXT =====\n")

        # 2. format prompt
        messages = prompt.format_messages(
            context=context_text,
            question=question
        )

        #3. call llm
        response = llm.invoke(messages)
        return response.content
    
    return rag_answer

def main():
    rag = build_rag_chain()
    print("Cricket CSV RAG Chatbot. Type 'exit' to quit.")
    while True:
        q = input("You: ")
        if q.lower() in ["exit","quit"]:
            break
        answer=rag(q)
        print("Bot: ",answer)

if __name__ == "__main__":
    main()
