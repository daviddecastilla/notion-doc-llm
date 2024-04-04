from config import settings
from notion_loader import NotionLoader
from langchain_community.document_loaders import NotionDBLoader


if __name__ == "__main__":
    notion_loader = NotionLoader(
        settings.notion_integration_token
    )

    docs = notion_loader.load()

    print(docs)