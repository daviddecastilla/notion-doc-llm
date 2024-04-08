# Notion LLM Project

This project is made to use retrieval augmented generation in order to 
use a notion workspace in combination with a LLM.

It uses Langchain to build the retrieval chain, and Ollama to
run models locally. 

## How to use ?

### Install
This project uses poetry as a python dependency manager. You can
learn how to use poetry [here](https://python-poetry.org/).

Another needed dependency is Ollama, that you can easily [install on 
your computer](https://ollama.com/).

### Configuration
For the configuration, the project is using [dynaconf](https://www.dynaconf.com/).
It means you can define the configuration with : 
- settings.toml file for non-sensitive variable
- .secrets.toml file for sensitive variable
- Env variable if you want to override what is in the two files.

> I encourage to learn how to use dynaconf as a configuration manager
> for python project on their website. 

The variables to define are : 
- `model`: the model to use, has to be a model available on Ollama, and already pulled on 
your computer (using `ollama pull <model>`)
- `retrieval.score_relevance_threshold`: The minimum relevance score
to retrieve document in the vector store
- `retrieval.max_doc_retrieved`: The max number of documents retrieved
from the vector store
- `notion_integration_token`: The notion integration token to access your notion
workspace (You need to share the pages with the integration once you created it).

> How to create a token : https://developers.notion.com/reference/create-a-token

### Run
```shell
poetry run python main.py
```
