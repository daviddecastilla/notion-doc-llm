import time

from config import settings
from notion_loader import NotionLoader

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

if __name__ == "__main__":
    loading_start = time.perf_counter()
    notion_loader = NotionLoader(
        settings.notion_integration_token
    )

    docs = notion_loader.load()

    embeddings = OllamaEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings, normalize_L2=True)

    loading_end = time.perf_counter()
    print(f'Loading and vectoring notion took {loading_end - loading_start}s')

    retrieval_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ('user', '{input}'),
        ("user",
         "Given the above conversation, generate a search query to look up to get information relevant "
         "to the conversation")
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions using the context provided below \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    llm_instanciation_start = time.perf_counter()
    llm = ChatOllama(model='mistral:instruct')
    llm_instanciation_end = time.perf_counter()

    print(f"Time to instantiate LLM : {llm_instanciation_end - llm_instanciation_end}s")

    retriever = vector.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.5, "k": 2}
    )

    retriever_chain = create_history_aware_retriever(
        llm, retriever, retrieval_prompt
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(
        retriever_chain, document_chain
    )

    chat_history = []

    # TODO : need to apply the Interface Segregation principle
    while True:
        message = input('Message : ')
        if message == 'exit':
            break

        response = retrieval_chain.invoke({
          "chat_history": chat_history,
          "input": message
        })

        print('AI : ')
        print(response['answer'])
        print('Context : ' + str(response['context']))
        chat_history.extend([
           HumanMessage(message),
           AIMessage(response['answer'])
        ])

