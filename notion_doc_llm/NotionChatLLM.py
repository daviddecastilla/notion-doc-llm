from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

from langchain_text_splitters import RecursiveCharacterTextSplitter

from notion_doc_llm.loader import NotionLoader


class NotionChatLLM:

    def __init__(self, notion_integration_token: str,
                 model: str = 'mistral:instruct',
                 relevance_score_threshold: float = 0.5,
                 max_doc_retrieved: int = 2):
        self.notion_integration_token = notion_integration_token
        self.model = model
        self.relevance_score_threshold = relevance_score_threshold
        self.max_doc_retrieved = max_doc_retrieved

        self.vector = self.load_data()
        self.chain = self.create_chain()

        self.chat_history = []

    def load_data(self):
        notion_loader = NotionLoader(
            self.notion_integration_token
        )

        docs = notion_loader.load()

        embeddings = OllamaEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings, normalize_L2=True)

        return vector

    def create_chain(self):
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

        llm = ChatOllama(model=self.model)

        retriever = self.vector.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': self.relevance_score_threshold,
                           "k": self.max_doc_retrieved}
        )

        retriever_chain = create_history_aware_retriever(
            llm, retriever, retrieval_prompt
        )

        document_chain = create_stuff_documents_chain(llm, prompt)

        retrieval_chain = create_retrieval_chain(
            retriever_chain, document_chain
        )

        return retrieval_chain

    def new_message(self, message: str):
        response = self.chain.invoke({
            "chat_history": self.chat_history,
            "input": message
        })

        self.chat_history.extend([
            HumanMessage(message),
            AIMessage(response['answer'])
        ])

        return response['answer'], response['context']
