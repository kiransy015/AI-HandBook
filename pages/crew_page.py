from crewai import Crew,Process
from pages.agents import blog_reseracher,blog_writer
from pages.tasks import research_task,write_task
import streamlit as st
import validators
import os

## Langcmith Tracking
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Langchain: Summarize Text From Youtube"

# Forming the tech-focused crew with some enhanced configuration
crew = Crew(
    agents=[blog_reseracher,blog_writer],
    tasks=[research_task,write_task],
    process=Process.sequential, # Optional: Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

## set up Streamlit
st.set_page_config(page_title="Langchain: Summarize Text From Youtube")
st.title("Langchain: Summarize Text From Youtube")
st.subheader('Summarize URL')

with st.expander("About this App"):
    st.write("""
        This application provides a visual dashboard for 
        Youtube Video Content Summarizer.
        
        **Tech Stack**:
        1. ChatOpenAI : Model - gpt-4o.
        2. Crew,Process.
        3. agents,tasks,tools.
        4. YoutubeChannelSearchTool.
        5. Streamlit App.

        **How to use**:
        1. Provide Youtube url.
             
        **Keywords to search**:
        1. https://www.youtube.com/watch?v=XoWveC7HwkE.
        2. https://www.youtube.com/watch?v=C_HzUyyVseI&t=36s.
    """)

user_input = st.text_input("Youtube URL:")

if not user_input.strip():
    st.error("Please provide the information to get started.")
elif not validators.url(user_input):
    st.error("Please enter a valid Youtube url.")
else:
    ## start the task execution process with enhanced feedack
    result = crew.kickoff(inputs={"topic":user_input})
    st.write(result.raw)
