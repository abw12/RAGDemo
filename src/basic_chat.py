from urllib import response
from langchain_community.chat_models import ChatOllama

def main():
    # 1) initializa locall llm via ollama
    llm = ChatOllama(model="llama3.1:8b",temperature=0.2)

    print("Cricket Chatbot (no RAG yet). Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit","quit"]:
            break

        response = llm.invoke(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()
    