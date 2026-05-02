# AI-Based Cardiovascular Disease Detection Using Human Lip Color

This project is a preliminary screening tool that utilizes Artificial Intelligence (AI) to detect potential signs of cardiovascular issues by analyzing the color of human lips.

## Overview
Cardiovascular diseases (CVDs) are a leading cause of fatalities globally. One of the physical indicators of poor oxygenated blood circulation—often linked to cardiovascular problems—is a bluish or purplish tint on the lips, known medically as **Cyanosis**. 

This system uses computer vision and machine learning techniques to:
1. Isolate the human lip region from a submitted image or webcam feed.
2. Extract the overall Hue, Saturation, and Value (HSV) color properties.
3. Automatically classify whether the tint falls into a "Normal" (healthy pink/red) or "High Risk" (cyanosis/blue) category.

## Project Structure
- `app.py`: The main Streamlit web application script providing an accessible UI.
- `src/lip_extractor.py`: Utility that leverages **Google's MediaPipe Face Mesh** to detect facial landmarks and isolate the outer lip bounds perfectly.
- `src/model_pipeline.py`: Contains a synthetic data generator (for prototype demonstration) and a `RandomForestClassifier` trained using **Scikit-learn** to infer cardiovascular risk based on RGB/HSV attributes.
- `requirements.txt`: Project dependencies.

## Prerequisites & Installation

Ensure you have Python 3.8+ installed on your system.

1. **Clone/Navigate** to the repository folder:
   ```bash
   cd LipDiseaseDetection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the Streamlit web application by running the following command:

```bash
streamlit run app.py
```

The interface will open in your default browser. You can:
- **Upload an image** from your local device.
- **Use your Webcam** to snap a self-portrait.

The platform will display the isolated lip cutout, your HSV analytics, and the AI diagnosis predicting the likelihood of cardiovascular risks based on lip color variations.

---
**Disclaimer:** *This application is exclusively designed as a preliminary screening and technological demonstration prototype. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.*
## How It Works
1. Detect face using MediaPipe FaceMesh
2. Identify lip region using facial landmarks
3. Extract lip color features (HSV/RGB)
4. Analyze color intensity and patterns
5. Classify into Normal or High Risk category

## Results
(Add screenshots here showing input image and output result)

## Future Improvements
- Improve dataset with real medical images
- Implement deep learning model (CNN)
- Add real-time webcam detection
- Enhance accuracy with better feature extraction

## Disclaimer
This project is for educational and research purposes only. It is not intended for medical diagnosis.
