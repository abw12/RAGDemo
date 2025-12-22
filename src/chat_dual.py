from pathlib import Path
from typing import Dict,Any

from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from rag_chat import BASE_DIR
from tabular_store import load_dataframes, find_player_names_in_question, get_player_rows

BASE_DIR = Path(__file__).parent.parent
VECTOR_DIR = BASE_DIR / "chroma_db"

# ---------- RAG SETUP ----------

def load_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(model="all-minilm")
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR)
    )
    return vectordb

def build_rag_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful cricket assistant. Use ONLY the given context "
            "from our stats CSVs to answer. If the answer is not evident, say "
            "'I don't find this clearly in my CSV knowledge base.'"
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}"
        )       
    ])

# ---------- STRUCTURED (PANDAS) TOOL ----------

def is_stats_query(question:str) -> bool:
    """
    Very simple heuristic router for now.
    If it mentions 'runs' or 'wickets' or 'average' AND 'player',
    we treat as stats query.
    You can refine this later or even use an LLM-based router.
    """  
    q = question.lower()
    has_metric = any(w in q for w in ["runs", "wickets", "average", "strike rate", "strike_rate"])
    # has_player = "player" in q or "batsman" in q or "bowler" in q
    # print(f"has_metric: {has_metric} and has_player: {has_player}")
    return has_metric

def structured_stats_answer(
        question: str,
        llm: ChatOllama,
        dfs: Dict[str,Any]
) -> Any:
    """
    1) Try to detect player_name from question.
    2) Look up rows in relevant DataFrames.
    3) Ask LLM to format answer.
    """
    # 1. Detect player names from data
    names = find_player_names_in_question(question,dfs,player_column="player_name")
    if not names:
        return "I could not match any player_name from the CSVs in your question. " \
               "Try including the exact player name as it appears in the data."
    
    if len(names) > 1:
        return (
            "Your question seems to reference multiple players: "
            + ", ".join(names)
            + ". Please specify exactly one player."
        )
    player_name = names[0]

    # 2. Get rows for that player from all tables
    tables = get_player_rows(dfs,player_name,player_column="player_name")
    if not tables:
        return f"I couldn't find any rows for player '{player_name}' in the CSVs."
    
    # Build a concise context text from these rows
    context_lines = []
    for table_name, df in tables.items():
        # For now, assume 'batter_player_stats' has total_runs etc.
        for _, row in df.iterrows():
            # Compact summary per row
            line = f"[{table_name}]" + ", ".join(f"{col}: {row[col]}" for col in df.columns)
            context_lines.append(line)
    raw_context = "\n".join(context_lines)

    # 3. Let LLM format the final answer, given precise numeric context
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a cricket statistics assistant. "
            "You are given raw CSV row data for one player. "
            "Carefully read it and answer the user's question with exact numbers, "
            "without guessing. If you are unsure, say so."
        ),
        (
            "human",
            "CSV rows for the player:\n{context}\n\nQuestion: {question}"
        ),
    ])

    message = prompt.format_messages(context=raw_context,question=question)
    response = llm.invoke(message)
    return response.content

# ---------- MAIN DUAL-MODE CHAT ----------

def build_dual_chat():
    llm = ChatOllama(model="llama3.1:8b",temperature=0.1)

    # RAG
    vectordb = load_vector_store()
    retriever = vectordb.as_retriever(search_kwargs={"k":10})
    rag_prompt = build_rag_prompt()

    # Tabular (structured)
    dfs = load_dataframes()

    def chat_answer(question: str):
        # 1) Check if it's a stats-like query
        if is_stats_query(question):
            print("[ROUTER] Using STRUCTURED (pandas) path")
            answer = structured_stats_answer(question,llm,dfs)
            if "I could not match any player_name" not in answer:
                return answer
            # else fall through to RAG as backup
        
        # 2) Fallback / general case: RAG over CSV text
        print("[ROUTER] Using RAG path")
        # Embed query with the same prefix used at indexing
        rag_query = "search_query: " + question
        docs = retriever.invoke(rag_query)

        # Debug: show what was retrieved
        print(f"[DEBUG] Retrieved {len(docs)} from RAG")
        for i,d in enumerate(docs[:3]):
            print(f"\n[DEBUG] Doc #{i+1}")
            print("[DEBUG] METADATA:", d.metadata)
            print("[DEBUG] CONTENT:", d.page_content[:200].replace("\n", " "))
        context = "\n\n".join(d.page_content for d in docs)
        messages = rag_prompt.format_messages(context=context,question=question)
        response = llm.invoke(messages)
        return response.content

    return chat_answer

def main():
    chat = build_dual_chat()
    print("Cricket Dual-Mode Chatbot (RAG + Structured). Type 'exit' to quit.")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break
        ans = chat(q)
        print("Bot:", ans)

if __name__ == "__main__":
    main()