from crewai.tools import tool
from langchain_community.document_loaders import YoutubeLoader

@tool("Generic URL Loader")
def yt_loader_tool(url: str) -> str:
    """
    Useful for loading and extracting content from any website or YouTube URL.
    Input should be a single string representing the URL.
    """
    # Initialize the loader with your specific generic headers
    loader = YoutubeLoader.from_youtube_url(url,add_video_info=False)

    # Load and return the text content
    data = loader.load()
    return data
