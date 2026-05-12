## RAG Q&A Conversation With TXT Including Chat History

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
st.title("Conversational RAG With TXT uploads and chat history")
st.write("Upload Txt's and chat with their content")

with st.expander("About this App"):
    st.write("""
        This application provides a visual dashboard for 
        Conversational RAG With TXT.
        
        **Tech Stack**:
        1. Groq , Model - llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-120b.
        2. HuggingFaceEmbeddings , Model - all-MiniLM-L6-v2.
        3. TextLoader.
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
        3. Upload Txt files.
        4. Submit Your Questions.
             
        **Keywords to search**:
        1. Element Not Found.
        2. Command Timeout.
        3. Appium Server Errors.
        4. Device Crash.
        5. Network Issues.
        6. Give timestamp and details of all session errors , separate each section.
        7. What all our previous conversation.
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

    session_id = st.text_input("Session ID",value="txt_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store = {}

    txt_files = st.file_uploader("Choose a TXT file",type="txt",accept_multiple_files=True)
    ## Process uploaded TXT's
    if txt_files:
        documents = process_uploaded_files(txt_files, 'txt')
        submit_your_questions(documents, embeddings,llm,session_id)

else:
    st.warning("Please enter the Groq APi Key")

