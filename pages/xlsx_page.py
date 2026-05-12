## RAG Q&A Conversation With XLSX Including Chat History

import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from utils.prepare_data import process_uploaded_files, submit_your_questions
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## set up Streamlit
st.title("Conversational RAG With XLSX uploads and chat history")
st.write("Upload Xlsx's and chat with their content")

with st.expander("About this App"):
    st.write("""
        This application provides a visual dashboard for 
        Conversational RAG With XLSX.
        
        **Tech Stack**:
        1. Groq , Model - llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-120b.
        2. HuggingFaceEmbeddings , Model - all-MiniLM-L6-v2.
        3. UnstructuredExcelLoader.
        4. RecursiveCharacterTextSplitter.
        5. Chroma Db.
        6. ChatPromptTemplate.
        7. session_id.
        8. create_stuff_documents_chain.
        9. BaseChatMessageHistory , ChatMessageHistory , RunnableWithMessageHistory.
        10. Streamlit App.
             
        **How to use**:
        1. Provide Groq API Key.
        2. Select Groq AI model.
        3. Upload Xlsx files.
        4. Submit Your Questions.
             
        **Keywords to search**:
        1. Give me the Physical Location of Employee Kiran.
        2. Who all are having iPhone device.
        3. Give me the dates where Kiran took Planned Leave.
        4. What all our previous conversation.
    """)

## Input the Groq API key
# api_key = st.text_input("Enter your Groq APi key:",type="password")

## Sidebar for settings
st.sidebar.title("Settings")
## Input the Groq API key
api_key = st.sidebar.text_input("Enter your Groq APi key:",type="password")

## Drop down to select various Groq AI models
model = st.sidebar.selectbox("Select an Groq AI model",["llama-3.3-70b-versatile","llama-3.1-8b-instant","openai/gpt-oss-120b"])

## Check if qroq api key is provided
if api_key:
    llm = ChatGroq(groq_api_key = api_key,model_name=model)

    ## chat interface

    session_id = st.text_input("Session ID",value="xlsx_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store = {}

    xlsx_files = st.file_uploader("Choose a XLSX file",type="xlsx",accept_multiple_files=True)
    # ## Process uploaded Xls's
    if xlsx_files:
        documents = process_uploaded_files(xlsx_files, 'xlsx')
        submit_your_questions(documents, embeddings,llm,session_id)

else:
    st.warning("Please enter the Groq APi Key")

