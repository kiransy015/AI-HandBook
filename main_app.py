import streamlit as st

# Define your pages
home_page = st.Page("pages/home.py", title="Home", icon=":material/home:",default=True)
pdf1_page = st.Page("pages/pdf1_page.py", title="PDF RAG Engine_Groq_ChromaDb", icon=":material/picture_as_pdf:")
xlsx_page = st.Page("pages/xlsx_page.py", title="XLSX RAG Engine_Groq_ChromaDb", icon=":material/csv:")
txt_page = st.Page("pages/txt_page.py", title="TXT RAG Engine_Groq_ChromaDb", icon=":material/text_snippet:")
web_page = st.Page("pages/web_page.py", title="WEB Search Engine_Groq", icon=":material/globe:")
db_page = st.Page("pages/db_page.py", title="DB Search Engine_Groq", icon=":material/database:")
website_page = st.Page("pages/website_page.py", title="Video & URL Summarizer_Groq", icon=":material/media_link:")
# pdf2_page = st.Page("pages/pdf2_page.py", title="PDF RAG Engine_OpenAI_AstraDb[ERROR]", icon=":material/picture_as_pdf:")
pdf3_page = st.Page("pages/pdf3_page.py", title="PDF RAG Engine_Nvidia_FAISSDb", icon=":material/picture_as_pdf:")
crew_page = st.Page("pages/crew_page.py", title="Video Content Summarizer Using Agents_OpenAI", icon=":material/media_link:")
hybridsearch_page = st.Page("pages/hybridsearch_page.py", title="Hybrid Search RAG_HuggingFace_Pinecone", icon=":material/database_search:")
imageCompare_page = st.Page("pages/imageCompare_page.py", title="AI Image Comparer_OpenAI", icon=":material/animated_images:")

# Create and run the navigation
pg = st.navigation([home_page,pdf1_page,xlsx_page,txt_page,web_page,db_page,website_page,pdf3_page,crew_page,hybridsearch_page,imageCompare_page])
pg.run()
