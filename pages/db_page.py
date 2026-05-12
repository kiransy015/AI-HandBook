import streamlit as st
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from utils.prepare_data import configure_db,db_datafetch

st.set_page_config(page_title="Langchain: Chat with SQL DB")
st.title("Langchain: Chat with SQL DB")

INJECTION_WARNING = """
    SQL agent can be vulnerable to prompt injection. Use a DB role with limited permissions.
    Read more [here](https://python.langchain.com/docs/security)
"""

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLLite 3 Database- student.db","Connect to you MySQL Database"]

selected_opt = st.sidebar.radio(label="Choose the DB which you want to chat",options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL password",type="password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_uri = LOCALDB

## Sidebar for settings
st.sidebar.title("Settings")
## Input the Groq API key
api_key = st.sidebar.text_input(label="Groq APi Key",type="password")

if not db_uri:
    st.info("Please enter the database information and uri")

if not api_key:
    st.info("Please add the groq api key")

with st.expander("About this App"):
    st.write("""
        This application provides a visual dashboard to 
        Chat with Database(Sqllite,MySql).
        
        **Tech Stack**:
        1. Groq , Model - llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-120b.
        2. AgentType,create_sql_agent.
        3. SQLDatabase,SQLDatabaseToolkit,create_engine.
        4. StreamlitCallbackHandler.
        5. Streamlit App.

        **How to use**:
        1. Choose the DB.
        2. Provide Groq API Key.
        3. Select Groq AI model.
        4. Submit Your Questions.
             
        **Keywords to search**:
        1. Fetch all records from database.
        2. Give me details of Student Kumar.
        3. Who all are in Data Science.
    """)

## Drop down to select various Groq AI models
model = st.sidebar.selectbox("Select an Groq AI model",["llama-3.3-70b-versatile","llama-3.1-8b-instant","openai/gpt-oss-120b"])

## LLM model
llm = ChatGroq(groq_api_key = api_key,model_name=model,streaming=True)

if db_uri==MYSQL:
    db = configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db = configure_db(db_uri)

## toolkit
toolkit = SQLDatabaseToolkit(db=db,llm=llm)
db_datafetch(llm,toolkit)
