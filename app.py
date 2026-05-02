import streamlit as st
from PIL import Image
import sys
import os

# Add src to the system path to allow straightforward imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.lip_extractor import LipExtractor
from src.model_pipeline import LipColorClassifier

# Configure Streamlit UI aesthetics
st.set_page_config(page_title="AI Cardiovascular Disease Detection", layout="wide", page_icon="🫀")

st.markdown("""
<style>
    .main-header {
        font-size: 38px;
        color: #1F2937;
        text-align: center;
        margin-bottom: 10px;
        font-family: sans-serif;
        font-weight: 700;
    }
    .sub-text {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 40px;
    }
    .risk-high {
        color: white;
        background-color: #EF4444;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .risk-low {
        color: white;
        background-color: #10B981;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Cache objects so model and mediapipe aren't rebuilt on every re-render
@st.cache_resource
def load_extractor():
    return LipExtractor()

@st.cache_resource
def load_model():
    # Store the model inside the project root
    model_file = os.path.join(os.path.dirname(__file__), "lip_model.pkl")
    return LipColorClassifier(model_path=model_file)

extractor = load_extractor()
classifier = load_model()

st.markdown('<div class="main-header">AI Cardiovascular Risk Detection via Lip Color 🫀</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload a facial image or use your webcam to analyze lip color indicators for potential cardiovascular risks (Cyanosis).</div>', unsafe_allow_html=True)

# Layout Setup
col1, col2 = st.columns([1, 1.2])

with col1:
    st.write("### 📸 Input Image")
    st.markdown('<div class="info-box">Please ensure you are in a well-lit environment and directly facing the camera.</div>', unsafe_allow_html=True)
    input_type = st.radio("Choose input method:", ("File Upload", "Webcam"), horizontal=True)
    
    img = None
    if input_type == "File Upload":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_container_width=True)
    else:
        captured_img = st.camera_input("Take a picture")
        if captured_img is not None:
            img = Image.open(captured_img)

with col2:
    st.write("### 🔬 Analysis & Results")
    
    if img is not None:
        with st.spinner('Analyzing facial landmarks and extracting lip color...'):
            raw_lip, masked_lip, mean_hsv, mean_rgb = extractor.extract_lips_and_color(img)
            
            if raw_lip is None:
                st.error("❌ No facial landmarks detected. Please ensure your face is clearly visible to the camera.")
            else:
                col_lip, col_stats = st.columns([1, 1])
                
                with col_lip:
                    st.write("#### Extracted Lip Region")
                    st.image(masked_lip, caption="Isolated Lips", width=200)
                    # Convert to ints
                    r, g, b = map(int, mean_rgb)
                    h, s, v = mean_hsv
                    
                with col_stats:
                    st.write("#### Color Analytics")
                    hex_color = '#%02x%02x%02x' % (r, g, b)
                    st.markdown(f"**Average Tint:**")
                    st.markdown(f'<div style="background-color: {hex_color}; width: 100px; height: 50px; border-radius: 5px; border: 1px solid #ccc; margin-bottom: 10px;"></div>', unsafe_allow_html=True)
                    st.write(f"- **RGB:** ({r}, {g}, {b})")
                    st.write(f"- **HSV:** ({h:.1f}, {s:.1f}, {v:.1f})")
                
                st.markdown("---")
                
                # Predict Risk
                st.write("#### AI Diagnosis")
                prediction, probs = classifier.predict(mean_hsv)
                confidence = probs[prediction] * 100
                
                if prediction == 1:
                    st.markdown('<div class="risk-high">⚠ HIGH RISK (Cyanosis Indicator)</div>', unsafe_allow_html=True)
                    st.write(f"> **Confidence:** {confidence:.1f}%")
                    st.error("The AI spotted bluish/purplish tints in the lip region which may denote reduced oxygen supply (Cyanosis/Hypoxia). We recommend consulting a healthcare professional.")
                else:
                    st.markdown('<div class="risk-low">✅ NORMAL (Healthy Lips)</div>', unsafe_allow_html=True)
                    st.write(f"> **Confidence:** {confidence:.1f}%")
                    st.success("The system detected standard red/pink lip coloration. No visible indicators of cyanosis were found.")

    else:
        st.info("Awaiting image input. Please select a file or take a picture using your webcam.")
