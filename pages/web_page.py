import streamlit as st
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
import os
from dotenv import load_dotenv
from utils.prepare_data import web_search

## Arxiv and wikipedia Tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchRun(name="Search")

tools = [arxiv,wiki,search]

st.title("Langchain - chat with search")
"""
In this example, we're using 'StreamlitCallbackHandler' to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more Langchain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
# st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Enter your Groq API Key:",type="password")

with st.expander("About this App"):
    st.write("""
        This application provides a visual dashboard for 
        Web Search.
        
        **Tech Stack**:
        1. Groq.
        2. Model - llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-120b.
        3. ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun.
        4. Agents,Tasks,Tools.
        5. StreamlitCallbackHandler.
        6. Streamlit App.

        **How to use**:
        1. Provide Groq API Key.
        2. Select Groq AI model.
        3. Submit Your Questions.
             
        **Keywords to search**:
        1. What is Machine Learning?.
        2. What is Generative AI?.
        3. arXiv:2604.28158.
    """)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assisstant","content":"Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

## Sidebar for settings
st.sidebar.title("Settings")
## Input the Groq API key
api_key = st.sidebar.text_input("Enter your Groq APi key:",type="password")

## Drop down to select various Groq AI models
model = st.sidebar.selectbox("Select an Groq AI model",["llama-3.3-70b-versatile","llama-3.1-8b-instant","openai/gpt-oss-120b"])


web_search(model,api_key,tools)

