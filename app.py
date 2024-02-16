import streamlit as st
import os
import google.generativeai as genai

from PIL import Image

# Use your API key
GOOGLE_API_KEY = "Use API key"

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure generative AI library
genai.configure(api_key=GOOGLE_API_KEY)

# Load Gemini Pro models
pro_model = genai.GenerativeModel("gemini-pro")
pro_vision_model = genai.GenerativeModel("gemini-pro-vision")

# Function to get Gemini Pro model response
@st.cache_data
def get_gemini_response(question):
    response = pro_model.generate_content(question)
    return response.text

# Function to get Gemini Pro Vision model response
@st.cache_data
def get_gemini_vision_response(_images, input):
    prompt = [input] + _images if input else _images
    response = pro_vision_model.generate_content(prompt)
    return response.text

# Set page configuration
st.set_page_config(page_title="Gemini PRO Model QnA", layout="wide")

# Sidebar
st.sidebar.title("Model Selection")
selected_model = st.sidebar.selectbox(
    "Which Model would you like to select",
    ("Gemini Pro Model", "Gemini Pro Vision Model")
)

# Toggle switch for st.cache
use_cache = st.sidebar.checkbox("Log data", value=True)

# Display different content based on the selected model
if selected_model == "Gemini Pro Model":
    st.header("Gemini Pro LLM Application")
    input_question = st.text_input("Prompt Here:", key="input")

    submit_button = st.button("Ask the Question")

    if submit_button:
        if use_cache:
            response = get_gemini_response(input_question)
        else:
            response = get_gemini_response(input_question)
        st.subheader("The Response is:")
        st.write(response)

elif selected_model == "Gemini Pro Vision Model":
    st.header("Gemini Pro Vision LLM Application for multiple images")
    input_question = st.text_input("Prompt Here:", key="input")

    uploaded_files = st.file_uploader("Choose Images(Upto 4 images)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    images = []

    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image {i + 1}", use_column_width=False, width=100)
            images.append(image)

    submit_button = st.button("Describe the given Images")

    if submit_button:
        response = get_gemini_vision_response(images, input_question)
        st.subheader("Image Descriptions:")
        st.write(response)
