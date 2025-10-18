import streamlit as st
import pandas as pd
import json
from streamlit_lottie import st_lottie
from Utils.rec_filter import rec_filter, fs
import os
import torch
torch.backends.quantized.engine = 'qnnpack'
# Page configuration
st.set_page_config(
    page_title="Medical Entity Extractor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load example cases
def load_example_cases():
    examples_path = os.path.join(current_dir, "Utils", "examples.txt")
    examples = []
    current_example = ""
    example_number = 1
    
    with open(examples_path, 'r') as file:
        for line in file:
            if line.strip().startswith(str(example_number) + "."):
                if current_example:
                    examples.append(current_example.strip())
                current_example = line[line.find(".") + 1:]
                example_number += 1
            else:
                current_example += line
                
    if current_example:
        examples.append(current_example.strip())
    
    return examples

# Function to load Lottie animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Get the current file's directory and load animation
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "Utils", "medical.json")
lottie_medical = load_lottiefile(json_path)

# Enhanced CSS with animations and modern styling
st.markdown("""
    <style>
    /* Main theme */
    .main {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling with gradient */
    h1 {
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 1rem;
        border-radius: 10px;
        background-color: #1e2433;
        color: white;
        border: 2px solid #2e3649;
        transition: all 0.3s ease;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 210, 255, 0.2);
    }
    </style>
""", unsafe_allow_html=True)


# App header with animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st_lottie(lottie_medical, height=250, key="medical")
    st.markdown("<h1>Medical Entity Extractor</h1>", unsafe_allow_html=True)

# Input section with description
st.markdown("""
    <div style='background-color: #1e2433; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h4 style='color: #00d2ff; margin-top: 0;'>📝 Enter Medical Text</h4>
        <p style='color: #ffffff99;'>Input your medical text below for entity extraction. Our advanced NLP system will:</p>
        <ul style='color: #ffffff99; margin-left: 1.5rem;'>
            <li>Extract and identify medical symptoms</li>
            <li>Determine symptom durations</li>
            <li>Map associated organs</li>
            <li>Provide detailed entity analysis</li>
        </ul>
        <p style='color: #ffffff99; font-style: italic; margin-top: 1rem;'>
            💡 The system uses state-of-the-art medical NLP to extract and analyze medical entities from your text.
        </p>
        <div style='background: linear-gradient(135deg, #3a7bd520, #00d2ff10); border: 1px solid #00d2ff; 
             border-radius: 8px; padding: 1rem; margin-top: 1rem;'>
            <p style='color: #00d2ff; margin: 0;'>
                ⚠️ First-time users: The application needs to download required model weights (~400MB) on initial startup. 
                This may take a few minutes depending on your internet connection. Please be patient while the models are being loaded.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Example cases section
st.markdown("""
    <div style='background-color: #1e2433; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h4 style='color: #00d2ff; margin-top: 0;'>📋 Example Cases</h4>
        <p style='color: #ffffff99;'>Click on any example case below to analyze it:</p>
    </div>
""", unsafe_allow_html=True)

# Load and display example cases
examples = load_example_cases()
cols = st.columns(3)
for idx, example in enumerate(examples):
    with cols[idx % 3]:
        if st.button(f"Example {idx + 1}", key=f"example_{idx}"):
            st.session_state.medical_text = example
            st.session_state.analyze_clicked = True

medical_text = st.text_area(
    label="Medical Text Input",
    height=150,
    placeholder="Enter your medical text here...",
    value=st.session_state.get('medical_text', ''),
    label_visibility="collapsed"
)

# Submit button with loading state
analyze_clicked = st.button("Analyze Text") or st.session_state.get('analyze_clicked', False)

if analyze_clicked:
    st.session_state.analyze_clicked = False  # Reset the flag
    with st.spinner('Extracting medical entities...'):
        # Only extract medical entities, no MRI prediction
        df1 = rec_filter(medical_text, fs)
        
        # Create DataFrame with medical entities only
        df = pd.DataFrame(df1, columns=['Entity', 'Duration', 'Type'])
        
        # Duration is already in text format, no conversion needed
        
        # Ensure all columns are strings for consistent display
        for col in df.columns:
            df[col] = df[col].astype(str)

    if not df.empty:
        # Results section with explanation
        st.markdown("""
            <div style='background-color: #1e2433; padding: 0.75rem; border-radius: 10px; margin-top: 2rem;'>
                <h4 style='color: #00d2ff; margin: 0;'>🔍 Analysis Results</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Display the DataFrame
        st.dataframe(df, height=400)

        # Add explanation as an expander
        with st.expander("ℹ️ Understanding the Results"):
            st.markdown("""
                ### Duration Field:
                - Shows the actual duration text as it appears in the original medical text
                - Examples: "2 hours", "3 days", "1 week", "6 months", etc.
                - "nil duration" indicates no specific duration was mentioned

                ### Entity Types:
                - **symptom**: Medical symptoms found in the text
                - **organ**: Body organs mentioned in the text
                
                ### Note:
                All entities are extracted independently without any relationships between them.
                Durations are preserved in their original text format.
            """)

        # Display summary statistics
        st.markdown("""
            <div style='background-color: #1e2433; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;'>
                <h4 style='color: #00d2ff; margin-top: 0;'>📊 Extraction Summary</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entities", len(df))
        with col2:
            symptoms_count = len(df[df['Type'] == 'symptom']) if not df.empty else 0
            st.metric("Symptoms Found", symptoms_count)
        with col3:
            organs_count = len(df[df['Type'] == 'organ']) if not df.empty else 0
            st.metric("Organs Found", organs_count)
        with col4:
            entities_with_duration = len(df[df['Duration'] != 'nil duration']) if not df.empty else 0
            st.metric("With Duration", entities_with_duration)
    else:
        st.warning("No medical entities were detected in the provided text. Please try again with different text.")

# Footer
st.markdown("""
    <div style='position: fixed; bottom: 0; left: 0; right: 0; background-color: #1e2433; 
    padding: 1rem; text-align: center; font-size: 0.8rem; color: #ffffff99;'>
        Made with ❤️ at RVCE
    </div>
""", unsafe_allow_html=True)
