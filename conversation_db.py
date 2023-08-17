from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

import os
os.environ["OPENAI_API_KEY"] = "...."

prompt_template = """Impersonate a tech assistant, Use the following pieces of context to give the solution the Problem. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}

Question: {question}
Answer in English:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

vectorstore = Chroma(persist_directory="store", embedding_function=OpenAIEmbeddings())

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())


user_problem = input("What is you issue? ")

solution = qa.run(user_problem)
print(solution)