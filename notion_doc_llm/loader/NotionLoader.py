import json
import requests

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_community.document_loaders import NotionDBLoader


class NotionLoader(BaseLoader):
    def __init__(self, notion_integration_token: str):
        self.api_url = "https://api.notion.com/v1/"
        self.headers = {
            'Notion-Version': '2022-06-28',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {notion_integration_token}'
        }

        self.notion_db_loader = NotionDBLoader(
            notion_integration_token, "database_id"
        )

    def load(self) -> list[Document]:
        pages = self._retrieve_pages_summary()

        return list(
            self.notion_db_loader.load_page(page_summary)
            for page_summary in pages
        )

    def _retrieve_pages_summary(self):
        pages = []
        query_dict = {}
        while True:
            response = requests.post(self.api_url + 'search',
                                     headers=self.headers,
                                     data=json.dumps(query_dict))

            pages.extend(response.json()['results'])

            if not response.json()['has_more']:
                break

            query_dict = {
                'start_cursor': response.json()['next_cursor']
            }

        return pages
