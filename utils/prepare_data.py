import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader, TextLoader
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_classic.agents import initialize_agent,AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from pathlib import Path
from langchain_classic.sql_database import SQLDatabase
from sqlalchemy import create_engine
import sqlite3
from langchain_classic.agents import create_sql_agent
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
import validators
from langchain_classic.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.vectorstores.cassandra import Cassandra
from langchain_classic.indexes.vectorstore import VectorStoreIndexWrapper 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def process_uploaded_files(uploaded_files, file_type):
    """Processes uploaded files and returns LangChain documents."""
    documents = []
    
    if not uploaded_files:
        return documents

    for uploaded_file in uploaded_files:
        # Create a temp file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            # Choose loader based on type
            if file_type == 'pdf':
                loader = PyPDFLoader(tmp_path)
            elif file_type == 'xlsx':
                loader = UnstructuredExcelLoader(tmp_path, mode="elements")
            elif file_type == 'txt':
                loader = TextLoader(tmp_path)
            else:
                continue
                
            docs = loader.load()
            documents.extend(docs)
            st.write(f"Processed {len(docs)} segments from {uploaded_file.name}")
        finally:
            # Ensure file is removed
            os.remove(tmp_path)
            
    return documents


def submit_your_questions(documents, embeddings,llm,session_id):
    ## Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. DO NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    ## Answer question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your questions:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config = {
                "configurable": {"session_id":session_id}
            }, # constructs a key "abc123" in 'store'.
        )

        st.write(st.session_state.store)
        st.write("Assistant:",response['answer'])
        st.write("Chat History:", session_history.messages)


def web_search(model,api_key,tools):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])

    if prompt:=st.chat_input(placeholder="What is Machine Learning?") :
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").write(prompt)

        llm = ChatGroq(groq_api_key = api_key,model_name=model,streaming=True)
        tools = tools

        search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = search_agent.run(prompt,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write(response)


@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri=="USE_LOCALDB":
        dbfilepath = (Path(__file__).parent.parent/"database"/"student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri=="USE_MYSQL":
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    

def db_datafetch(llm,toolkit):
    agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role":"assistant","content":"How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask anything from the database")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(user_query,callbacks=[streamlit_callback])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write(response)


def Summarizer_websiteContent(llm,groq_api_key,generic_url):
    prompt_template = """
        Provide a summary of the following content in 300 words:
        Content:{text}
    """

    prompt = PromptTemplate(template=prompt_template,input_variables=["text"])

    if st.button("Summarize the Content from YT or Website"):
        ## Validate all the inputs
        if not groq_api_key.strip() or not generic_url.strip():
            st.error("Please provide the information to get started")
        elif not validators.url(generic_url):
            st.error("Please enter a valid url. It can be YT video or website url")
        else:
            try:
                with st.spinner("Waiting..."):
                    ## loading the website or yt video date
                    if "youtube.com" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url,add_video_info=False)
                    else:
                        loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs = loader.load()

                    ## Chain for Summarization
                    chain = load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                    output_summary = chain.run(docs)

                    st.success(output_summary)
            except Exception as e:
                print(e)              # message
                import traceback
                traceback.print_exc() # full stack trace


def upload_ToAstraDb(raw_text,embeddings):
    #### Create your LangChain vector store.... backend by Astra DB! 
    astra_vector_store = Cassandra( 
        embedding=embeddings, 
        table_name="qa_mini_demo", 
        session=None, 
        keyspace=None 
    ) 

    from langchain_classic.text_splitter import CharacterTextSplitter 
    # We need to split the text using Character Text Split such that it should not increase token size 
    text_splitter = CharacterTextSplitter( 
        separator = '\n', 
        chunk_size = 800, 
        chunk_overlap = 200, 
        length_function = len, 
    ) 

    texts = text_splitter.split_text(raw_text) 
    texts[:50] 

    #### Load the dataset into the vector store 
    astra_vector_store.add_texts(texts[:50]) 
    print("Inserted %i headline." % len(texts[:50])) 
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    return astra_vector_store,astra_vector_index


def dataFetch_FromAstraDb(llm,astra_vector_store,astra_vector_index,user_input):
    print("\nQUESTION: \"%s\"" % user_input) 
    answer = astra_vector_index.query(user_input, llm=llm).strip() 
    print("ANSWER: \"%s\"\n" % answer) 

    print("FIRST DOCUMENTS BY RELEVANCE:") 
    for doc, score in astra_vector_store.similarity_search_with_score(user_input, k=4): 
        print("    [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))


@st.cache_resource(ttl="2h")
def uploadEmbeddings_ToFAISSDb():
    if "vectors" not in st.session_state:
        filepath = (Path(__file__).parent.parent/"uploads"/"us_census").absolute()
        print(filepath)
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader(filepath)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50,) ## Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


def dataFetch_FromFAISSDb(llm,user_input):
    prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
        {context}
    <context>
    Question:{input}
    """   
    )

    if "vectors" in st.session_state and "retrieval_chain" not in st.session_state:
        document_chain = create_stuff_documents_chain(llm,prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
        st.session_state.retrieval_chain = create_retrieval_chain(retriever,document_chain)

    if user_input and "retrieval_chain" in st.session_state:
        response = st.session_state.retrieval_chain.invoke({"input":user_input})
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i , doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("---------------------------------")
