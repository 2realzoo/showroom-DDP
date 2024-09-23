from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain.embeddings import OllamaEmbeddings, OpenAIEmbeddings

gpt_model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
llama_model = ChatOllama(temperature=0, model='llama3.1')
llama_embedding = OllamaEmbeddings()
gpt_embedding = OpenAIEmbeddings()