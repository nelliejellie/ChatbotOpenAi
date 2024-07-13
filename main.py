import streamlit as st
# stremlit helps tp create good web Ui
from PyPDF2 import PdfReader
# for reading pdfiles
from langchain.text_splitter import RecursiveCharacterTextSplitter
# for spliiting text into smaller chunks
from langchain.embeddings.openai import OpenAIEmbeddings
# using this library to upload our chunks of data to be embedded
from langchain.vectorstores import FAISS
import openai
import os
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


openai.api_key=os.getenv("OPENAI_API_KEY")



#upload a pdf file
st.header(openai.api_key)
st.title("Your documents")
file = st.file_uploader("upload pdf and start asking questions", type="pdf")

# read the pdf file
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for pages in pdf_reader.pages:
        text+=pages.extract_text()
        #st.write(text)

    # break it into chunks
    text_split = RecursiveCharacterTextSplitter(separators="\n", chunk_size=100,chunk_overlap=50,length_function=len)

    # initializing chunks
    chunks = text_split.split_text(text)
    #st.write(chunks)

    #generate embeddings
    # embedding using OpenAI

    # create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

    # store chunks and embeddings
    #creating a vector store using FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    #get user question
    user_question = st.text_input("Type your questions here")

    #do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        st.write(match)

        llm = ChatOpenAI(openai_api_key=openai.api_key, temperature=0.1,max_tokens=500,model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question=user_question)
        st.write(response)



