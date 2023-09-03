import streamlit as st
from PyPDF2 import PdfReader
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import pandas as pd
from langchain.memory import ConversationBufferMemory
# from io import StringIO
from langchain.prompts import PromptTemplate
import os
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain
# from streamlit.state.session_state import get_state
# print('test')
# if 'ans' not in st.session_state:
st.title('Doc QA')
os.environ["OPENAI_API_KEY"] = "Input you API Key"
# memory = ConversationBufferMemory(memory_key="history")
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key = 'history')

upload_file = st.file_uploader('Choose a file:')
# print('file uploaded')
if upload_file:
    reader = PdfReader(upload_file)
    # print(reader)

    # raw_text = state.raw_text or ''
    raw_text = st.session_state.get('raw_text', '')
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # print(raw_text)

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)


    # print(len(texts))

    # print(texts)
    # print('text split')
    embeddings = OpenAIEmbeddings()
    

    docsearch = FAISS.from_texts(texts, embeddings)


    # print('docsearch')
    query_unq = st.text_input('Enter your query')



    

    if query_unq:

        role_template = '''
        {query_unq}: based on this query, answer in one word who do you think would be an expert to solve it? Dont give the solution
        of the statement, only give the expert
        '''
        role_prompt = PromptTemplate(
            input_variables=["query_unq"],
            template=role_template,
        )
        # chain = load_qa_chain(OpenAI(temperature = 0.1), chain_type="stuff")
        role_query = role_prompt.format(query_unq = query_unq)
        docs = docsearch.similarity_search(query_unq)
        # print(len(docs))
        # role_memory = ConversationBufferMemory(input_key = 'role', memory_key = 'chat_history')
        role_chain = load_qa_chain(OpenAI(temperature = 0.1), chain_type="stuff")

        role = (role_chain.run(input_documents=docs, question=role_query))
        st.write(role)
        # print(role)

        ans_template = '''
        Now acting as {role}. 
        Based on the query, read through the document to answer the question appropriately.
        query is as follows: {query_unq}
        '''
        ans_prompt = PromptTemplate(
            input_variables = ['role', 'query_unq'],
            template = ans_template
        )
        # print(docs)


        ans_query = ans_prompt.format(role = role, query_unq = query_unq)
        # print(docs[0])
        # ans_memory = ConversationBufferMemory(input_key = 'question', memory_key = 'chat_history')
        memory = ConversationBufferMemory( memory_key="history")

        ans_chain = load_qa_chain(OpenAI(temperature = 0.1), chain_type="stuff", memory = memory)
        # ans_chain = ConversationChain(llm = OpenAI(temperature = 0.1), memory = st.session_state.memory)


        # print(memory)
        ans = ans_chain.run(input_documents = docs, question = ans_query)
        st.write(ans)

        