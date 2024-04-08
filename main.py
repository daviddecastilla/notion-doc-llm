from config import settings

from notion_doc_llm import NotionChatLLM

if __name__ == "__main__":

    chat_llm = NotionChatLLM(
        settings.notion_integration_token,
        settings.model,
        settings.retrieval.score_relevance_threshold,
        settings.retrieval.max_doc_retrieved
    )

    while True:
        message = input("Message : ")

        if message == 'exit':
            break

        answer, context = chat_llm.new_message(message)

        print(f"AI : {answer}")
        print(f"Context : {context}")
