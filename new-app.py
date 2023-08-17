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

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'
)

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
{context}
</ctx>
------
<hs>
{chat_history}
</hs>
------
You should ask the user to provide inputs (names written in <> brackets) if required when you are returning a command response of question:
{question}
for example:
Question: write a linux command to copy a file
Answer:  cp <from> <to>, please provide value of <from> and <to>
Question: write a command to get pod logs 
Answer:  kubectl logs <podName>, please provide the value of <podName> 
Question: write a command to ssh into a VM
Answer:  ssh@<userName> <IP>, please provide the value of <userName> and <IP>
Answer:
"""
PROMPT = PromptTemplate(
    input_variables=[ "chat_history","context", "question"],
    template=template,
)

question_generator_chain = LLMChain(llm=AzureOpenAI(
    openai_api_type = "azure",
    openai_api_version = "2023-05-15",
    openai_api_base = "https://onehackopenai-westeurope.openai.azure.com",
    engine="gpt-35-turbo",
    model="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=1000,
), prompt=PROMPT)

qa = ConversationalRetrievalChain.from_llm(
    AzureOpenAI(
        openai_api_type = "azure",
        openai_api_version = "2023-05-15",
        openai_api_base = "https://onehackopenai-westeurope.openai.azure.com",
        engine="gpt-35-turbo",
        model="gpt-3.5-turbo",
        temperature=0.3,
    ),
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": PROMPT},
    memory=memory
)

while True:
    query = input("Ask question! ")
    answer = qa.run(query)
    print("answer: "+ answer)

