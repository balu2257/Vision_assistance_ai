# VisionAssistant AI
import streamlit as st
from PIL import Image
import pyttsx3
import os
import pytesseract  
import cv2
import numpy as np
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
import logging
from ultralytics import YOLO


pytesseract.pytesseract.tesseract_cmd = r'"C:\Program Files\Tesseract-OCR\tesseract.exe"'



# Initialize Google Generative AI with API Key
llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key='AIzaSyCs60qWMnNF0e6NNgqEyQ8bT_4F8ACxSH4')

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Streamlit page configuration
st.markdown(
    """
    <style>
     .main-title {
        font-size: 48px;
        font-weight: 600;
        text-align: center;
        color: #4A90E2;
        margin-top: 0px;
    }
    .subtitle {
        font-size: 22px;
        color: #616161;
        text-align: center;
        margin-bottom: 30px;
    }
    .feature-header {
        font-size: 20px;
        color: #4A90E2;  /* Set heading color to blue like the main title */
        font-weight: bold;
    }
    .footer-text {
        text-align: center;
        font-size: 12px;
        color: #757575;
    }
    .upload-header {
        font-size: 24px;
        font-weight: 600;
        text-align: center;
        color: #4A90E2;  /* Set upload section header color to blue */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title"> VisionAssistant AI </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"> Leveraging AI to Aid the Visually Impaired in Their Daily Lives </div>', unsafe_allow_html=True)

# Sidebar Features
st.sidebar.image(
    r"https://blogs.opentext.com/wp-content/uploads/2023/02/IBM-meritless-claim-1200x765px-780x470.jpg",  
    width=250
)

st.sidebar.title("📝 Overview")
st.sidebar.markdown(
    """
    🌟 **Features**:
    - 👁️ **Scene Analysis**: AI-powered analysis to describe the image uploaded.
    - ✏️ **Text Extraction**: Optical Character Recognition (OCR) for extracting readable text from images.
    - 🔊 **Speech Output**: Converts extracted text into voice.
    - 🖼️ **Object Detection**: Detects and labels objects in the uploaded image.

    🌍 **Purpose**:
    Designed for visually impaired individuals, this tool provides image understanding, text extraction, and audio feedback.

    ⚙️ **Technology**:
    - **Google Gemini API** for scene comprehension.
    - **Tesseract OCR** for text extraction.
    - **pyttsx3** for audio output.
    - **YOLO Object Detection** for object recognition.
    """
)

# Instructions Section
st.sidebar.markdown(
    """
    ## 🚀 How to Use:
    1. **Upload an Image**: Click on the 'Upload Image' button to select an image file (JPG, JPEG, or PNG) from your device.
    2. **Select a Feature**:
       - **Scene Analysis**: Click on 'Analyze Scene' to receive a detailed description of the image, including objects and their purpose.
       - **Extract Text**: Click on 'Extract Text' to retrieve any visible text from the image using Optical Character Recognition (OCR).
       - **Hear the Text**: Click on 'Hear the Text' to listen to the extracted text aloud.
       - **Detect Objects**: Click on 'Detect Objects' to identify and label objects in the uploaded image.
    3. **Listen to the Output**: After selecting a feature, the result will be displayed on the main page, and you can listen to any text output.
    4. **Enjoy!**: This tool is designed to assist visually impaired users by providing essential information in audio and visual formats.
    """
)

# Functions for functionality
def extract_text_from_image(image):
    """Use OCR to extract any visible text from the image."""
    return pytesseract.image_to_string(image)

def text_to_speech(text):
    """Read aloud the provided text using the TTS engine."""
    engine.say(text)
    engine.runAndWait()

def generate_scene_description(input_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

def prepare_image(uploaded_file):
    """Convert uploaded file into image data for further processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")

def detect_objects_in_image(image):
    """Detect objects in the uploaded image using YOLOv3."""
    # Load YOLO model and classes
    model = YOLO("best (1).pt")
    if uploaded_file is not None:
    # Open the uploaded image
           
    
    # Display the uploaded image
       
    
    # Perform detection
       results = model(image, conf=0.5)  # Only show results with confidence > 0.5

    
    
       bbox_image = results[0].plot()  
    

       return st.image(bbox_image, caption='Tree Speices and leaf Disease checked', use_column_width=True)

# Upload Image Section
st.markdown("<div class='upload-header'>🔄 Upload Your Image Here</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Uploaded Image", use_column_width=True)

# Feature Selection Section (4 columns now)
st.markdown("<h3 class='feature-header'>✨ Select a Feature to Use</h3>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

scene_button = col1.button("👁️ Analyze Scene")
ocr_button = col2.button("✏️ Extract Text")
tts_button = col3.button("🔊 Hear the Text")
object_detect_button = col4.button("🖼️ Detect Objects")

# Store extracted text using session state
if 'extracted_text' not in st.session_state:
    st.session_state['extracted_text'] = None

if uploaded_file:
    if scene_button:
        # Prepare prompt and call scene description
        scene_description_prompt = """
        As an AI assistant, you are helping visually impaired users understand the contents of an image. Please:
        1. List objects in the image and their purpose.
        2. Provide a detailed description of the image.
        3. Suggest actions or safety measures for users with visual impairments.
        """
        image_data = prepare_image(uploaded_file)
        description = generate_scene_description(scene_description_prompt, image_data)
        st.markdown(f"**Scene Description:** {description}")

    if ocr_button:
        # Extract text from the image and display
        st.session_state.extracted_text = extract_text_from_image(image)
        st.markdown(f"**Extracted Text:** {st.session_state.extracted_text}")
    
    if tts_button:
        if st.session_state.extracted_text:
            text_to_speech(st.session_state.extracted_text)
        else:
            st.warning("Please extract text before using the text-to-speech feature.")

    if object_detect_button:
        detected_img = detect_objects_in_image(image)
        st.markdown("<h3 class='feature-header'>🖼️ Object Detection</h3>", unsafe_allow_html=True)
        

# Footer
st.markdown(
    """
    <hr>
    <footer class="footer-text">
        <p>Created using Streamlit | Powered by Google Gemini API</p>
        <p>Developer: Balakrishna Bode</p>
        <p>Contact: <a href="mailto:bodebalakrishna@gmail.com">bodebalakrishna@gmail.com</a> | 
        LinkedIn: <a href="www.linkedin.com/in/bala-krishna-bode">BalaKrishna Bode</a> | 
        GitHub: <a href="https://github.com/balu2257">Balakrishna Bode</a></p>
    </footer>
    """,
    unsafe_allow_html=True,
)