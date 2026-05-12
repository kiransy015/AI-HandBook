from crewai import Agent , LLM
from pages.tools import yt_loader_tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"
llm = "gpt-4o"

## Create a senior blog content researcher

blog_reseracher = Agent(
    role = 'Blog Researcher from Youtube Videos',
    goal= 'get the relevant video content from the Youtube Url {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science , Machine Learning And GEN AI and providing suggestion"
    ),
    tools=[yt_loader_tool],
    llm = llm,
    allow_delegation=True
)

## creating a senior blog writer agent with YT tool

blog_writer = Agent(
    role='Blog Writer',
    goal="Narrate compelling tech stories about the content from the Youtube Url {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[yt_loader_tool],
    llm = llm,
    allow_delegation=False
)




