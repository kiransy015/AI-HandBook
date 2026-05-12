import streamlit as st
import base64
from PIL import Image
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from streamlit_image_comparison import image_comparison

# Setup Page
st.set_page_config(page_title="AI Image Compare", layout="wide")
st.title("🖼️ AI Image Comparison & Analysis")

with st.expander("About this App"):
    st.write("""
        This application provides a visual dashboard for 
        Image Comparision.
        
        **Tech Stack**:
        1. ChatOpenAI , Models - gpt-5.4-mini,gpt-5.4-nano,gpt-4o.
        2. HumanMessage.
        3. Image,base64,BytesIO.
        4. image_comparison.
        5. Streamlit App.

        **How to use**:
        1. Provide OpenAI API Key.
        2. Select OpenAI model.
        3. Upload Images.
        4. Click on Analyze Button.
    """)

# Sidebar for API Key
# api_key = st.sidebar.text_input("OpenAI API Key", type="password")

## Sidebar for settings
st.sidebar.title("Settings")
## Input the OpenAI API key
api_key = st.sidebar.text_input("Enter your OpenAI APi key:",type="password")

## Drop down to select various OpenAI models
model = st.sidebar.selectbox("Select an OpenAI model",["gpt-5.4-mini","gpt-5.4-nano","gpt-4o"])

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# 1. File Uploaders
col1, col2 = st.columns(2)
with col1:
    img1_file = st.file_uploader("Upload Image A", type=["png", "jpg", "jpeg"])
with col2:
    img2_file = st.file_uploader("Upload Image B", type=["png", "jpg", "jpeg"])

if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    # 2. Visual Comparison Slider
    st.subheader("Visual Comparison")
    image_comparison(img1=img1, img2=img2, label1="Image A", label2="Image B")

    # 3. LangChain AI Analysis
    if api_key:
        if st.button("Analyze Differences with AI"):
            llm = ChatOpenAI(model=model, openai_api_key=api_key)
            
            # Prepare images for LangChain
            base64_img1 = encode_image(img1)
            base64_img2 = encode_image(img2)

            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Compare these two images. List the key differences and similarities in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img2}"}},
                ]
            )

            with st.spinner("AI is thinking..."):
                response = llm.invoke([message])
                st.subheader("AI Insight")
                st.write(response.content)
    else:
        st.warning("Please enter your OpenAI API key in the sidebar to use the AI analysis feature.")
