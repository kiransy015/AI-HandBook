## RAG Q&A Conversation With PDF Including Chat History

import streamlit as st
from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
from pinecone import Pinecone , ServerlessSpec
index_name = "hybrid-search-langchain-pinecone"
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from utils.prepare_data import process_uploaded_files
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

## set up Streamlit
st.title("Hybrid Search Conversational RAG With PDF uploads and chat history")
st.write("Upload Pdf's and chat with their content")

with st.expander("About this App"):
    st.write("""
        This application provides a visual dashboard for 
        Hybrid Search Conversational RAG With PDF.
        
        **Tech Stack**:
        1. HuggingFaceEmbeddings : Model - all-MiniLM-L6-v2,all-MiniLM-L12-v2.
        2. Pinecone,PineconeHybridSearchRetriever.
        3. BM25Encoder,RecursiveCharacterTextSplitter.
        4. Streamlit App.

        **How to use**:
        1. Provide Pinecone Key.
        2. Select HuggingFaceEmbeddings model.
        3. Upload Pdf files.
        4. Submit Your Questions.
             
        **Keywords to search**:
        1. How to become a Friendlier Person.
        2. How to be a Leader.
        3. What all our previous conversation.
    """)

## Input the Pinecone key
# api_key = st.text_input("Enter your Pinecone key:",type="password")

## Sidebar for settings
st.sidebar.title("Settings")
## Input the Pinecone API key
api_key = st.sidebar.text_input("Enter your Pinecone key:",type="password")
index_name = "langchain-pinecone-hybrid-search"

## Drop down to select various HuggingFaceEmbeddings models
model = st.sidebar.selectbox("Select an HuggingFaceEmbeddings model",["all-MiniLM-L6-v2","all-MiniLM-L12-v2"])
embeddings = HuggingFaceEmbeddings(model_name = model)

## Check if Pinecone api key is provided
if api_key:
    ## initialize the Pinecone client
    pc = Pinecone(api_key=api_key)
    
    ## chat interface
    session_id = st.text_input("Session ID",value="pdf_session")
    
    ## statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    pdf_files = st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)
    
    if pdf_files:
        documents = process_uploaded_files(pdf_files, 'pdf')

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Extract text content from the splits
        split_texts = [doc.page_content for doc in splits]

        #create the index
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384, # dimensionality of dense model
                metric="dotproduct", # sparse values supported only for dotproduct
                spec = ServerlessSpec(cloud="aws",region="us-east-1"),
            )
        index = pc.Index(index_name)

        bm25_encoder = BM25Encoder().default()

        ## tfidf values on these sentences
        bm25_encoder.fit(split_texts)

        ## store the values to a json file
        bm25_encoder.dump("bm25_values.json")

        # load to your BM25Encoder object
        bm25_encoder = BM25Encoder().load("bm25_values.json")

        retreiver = PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index)

        retreiver.add_texts(
            split_texts
        )

        user_input = st.text_input("Your questions:")
        if user_input:
            response = retreiver.invoke(user_input)
            for res in response:
                st.write(res.page_content)
                st.write("===================================")
        else:
            st.info("Please enter your questions")

        
else:
    st.warning("Please enter the Pinecone Key")

