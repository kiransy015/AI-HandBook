import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from utils.prepare_data import Summarizer_websiteContent


## streamlit App
st.set_page_config(page_title="Langchain: Summarize Text From YT or Website")
st.title("Langchain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

with st.expander("About this App"):
    st.write("""
        This application provides a visual dashboard for 
        Video & URL Content Summarizer.
        
        **Tech Stack**:
        1. Groq - Model - llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-120b.
        2. YoutubeLoader,UnstructuredURLLoader,validators.
        3. PromptTemplate,load_summarize_chain.
        4. Streamlit App.

        **How to use**:
        1. Provide Groq API Key.
        2. Select Groq AI model.
        3. Provide Youtube or Website url.
             
        **Keywords to search**:
        1. https://www.youtube.com/watch?v=XoWveC7HwkE.
        2. https://docs.langchain.com/langsmith/home.
    """)

## Get the Groq API Key and url(YT or website) to be summarized
with st.sidebar:
    st.title("Settings")
    groq_api_key = st.text_input("Groq API Key",value="",type="password")

generic_url = st.text_input("URL",label_visibility="collapsed")

## Drop down to select various Groq AI models
model = st.sidebar.selectbox("Select an Groq AI model",["llama-3.3-70b-versatile","llama-3.1-8b-instant","openai/gpt-oss-120b"])

## LLM model
llm = ChatGroq(groq_api_key = groq_api_key,model_name=model)

Summarizer_websiteContent(llm,groq_api_key,generic_url)

