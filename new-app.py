import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ( RetrievalQA, ConversationalRetrievalChain, LLMChain)
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
import os
memory = ConversationBufferMemory(memory_key="history",input_key="query" ,output_key='answer',return_messages=True)

vectorstore = Chroma(
    persist_directory="store", 
    embedding_function=OpenAIEmbeddings(
    openai_api_type = "azure",
    openai_api_version = "2023-05-15",
    openai_api_base = "https://onehackopenai-westeurope.openai.azure.com",
    chunk_size= 200,
))
template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
You are a software developer bot.
You must never speculate or provide incorrect answers.
You should respond to user queries concisely and accurately.
You should prompt the user for any necessary command inputs when you are returning a response
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=[ "history","context", "question"],
    template=template,
)

chat_template = ChatPromptTemplate.from_messages([
    """You are a software developer bot.
You must never speculate or provide incorrect answers.
You should respond to user queries concisely and accurately.
You should prompt the user for any necessary command inputs when you are returning a response"""
])

messages = chat_template

question_generator_chain = LLMChain(llm=AzureOpenAI(
    openai_api_type = "azure",
    openai_api_version = "2023-05-15",
    openai_api_base = "https://onehackopenai-westeurope.openai.azure.com",
    engine="gpt-35-turbo",
    model="gpt-3.5-turbo",
    temperature=0.0,
    max_tokens=1000,
), prompt=prompt)

qa = RetrievalQA.from_chain_type(
    llm=AzureOpenAI(
        openai_api_type = "azure",
        openai_api_version = "2023-05-15",
        openai_api_base = "https://onehackopenai-westeurope.openai.azure.com",
        engine="gpt-35-turbo",
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=1000,
    ),
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    verbose=False,
    chain_type_kwargs={
        "verbose": False,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
)

while True:
    query = input("Ask question! ")
    answer = qa.run(query)
    print("answer: "+ answer)

