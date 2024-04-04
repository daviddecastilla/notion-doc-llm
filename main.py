from config import settings
from notion_loader import NotionLoader

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

if __name__ == "__main__":
    notion_loader = NotionLoader(
        settings.notion_integration_token
    )

    docs = notion_loader.load()

    embeddings = OllamaEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ('user', '{input}'),
        ("user",
         "Given the above conversation, generate a search query to look up to get information relevant "
         "to the conversation")
    ])

    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "You are talking to David de Castilla, answer its questions based on the "
                   "below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    llm = Ollama(model='mistral')

    document_chain = create_stuff_documents_chain(llm, prompt2)

    retriever = vector.as_retriever()

    retriever_chain = create_history_aware_retriever(
        llm, retriever, prompt
    )

    chat_history = []

    retrieval_chain = create_retrieval_chain(
        retriever_chain, document_chain
    )

    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "How old am I ?"
    })

    print(response['context'])
    print(response['answer'])
