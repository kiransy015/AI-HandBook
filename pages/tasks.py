from crewai import Task
from pages.tools import yt_loader_tool
from pages.agents import blog_reseracher , blog_writer

## Research Task
research_task = Task(
    description=(
        "Get detailed information from the Youtube Url {topic}."
    ),
    expected_output='A comprehensive 3 paragraphs long report based on the video content from the Youtube Url {topic}',
    tools=[yt_loader_tool],
    agent=blog_reseracher,
)

#Writing task with language model configuration
write_task = Task(
    description=(
        "get the info from the Youtube Url {topic}."
    ),
    expected_output='Summarize the info from the Youtube Url {topic} and create the content for the blog',
    tools=[yt_loader_tool],
    agent=blog_writer,
    async_execution=False,
    output_file='new-blog-post.md' # Example of output customization
)

