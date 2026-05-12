import streamlit as st
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from utils.prepare_data import uploadEmbeddings_ToFAISSDb,dataFetch_FromFAISSDb

from dotenv import load_dotenv
load_dotenv()

## Load the Nvidia API Key
# os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
# llm = ChatNVIDIA(model="google/gemma-4-31b-it") ## NVIDIA NIM Inferencing

## Sidebar for settings
st.sidebar.title("Settings")
## Input the Nvidia API key
api_key = st.sidebar.text_input("Enter your Nvidia APi key:",type="password")

## Drop down to select various Nvidia AI models
model = st.sidebar.selectbox("Select an Nvidia AI model",["None","google/gemma-4-31b-it"])
llm = ChatNVIDIA(model=model) ## NVIDIA NIM Inferencing

st.title("Nvidia NIM Demo")

with st.expander("About this App"):
    st.write("""
        This application provides a visual dashboard for 
        Conversational RAG With PDF.
        
        **Tech Stack**:
        1. NVIDIAEmbeddings,ChatNVIDIA : Model - google/gemma-4-31b-it.
        2. PyPDFDirectoryLoader,RecursiveCharacterTextSplitter.
        3. ChatPromptTemplate,create_stuff_documents_chain,create_retrieval_chain.
        4. FAISS.
        5. Streamlit App.

        **How to use**:
        1. Provide Nvidia API Key.
        2. Select Nvidia AI model.
        3. Load the Embeddings.
        4. Submit Your Questions.
             
        **Keywords to search**:
        1. Give me the American Community Survey details.
        2. What all are Terms and Definitions.
    """)

if st.button("Document Embedding"):
    uploadEmbeddings_ToFAISSDb()
    st.write("FAISS Vector Store DB Is Ready Using NvidiaEmbedding")
    
user_input = st.text_input("Enter Your Question From Documents")

if user_input:
    dataFetch_FromFAISSDb(llm,user_input)


