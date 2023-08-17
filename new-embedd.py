import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ( RetrievalQA, ConversationalRetrievalChain, LLMChain)
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

directory_path = 'documents/'

# Get a list of all files in the directory
file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]


for file_name in file_list:
    print("Loading file: "+ file_name)
    loader = UnstructuredFileLoader(directory_path+file_name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(
        openai_api_type = "azure",
        openai_api_version = "2023-05-15",
        openai_api_base = "https://onehackopenai-westeurope.openai.azure.com",
        chunk_size= 1,
    )
    db = Chroma.from_documents(texts, embeddings, persist_directory="store")