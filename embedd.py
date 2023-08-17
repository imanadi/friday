from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

import os
api_key = os.getenv("OPENAI_API_KEY")
file_location = "text-docs/file.txt"

with open(file_location) as f:
    file = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function = len)
texts = text_splitter.split_text(file)

vectorstore = Chroma.from_texts(texts=texts, embedding=OpenAIEmbeddings(), persist_directory="store")
vectorstore.persist()