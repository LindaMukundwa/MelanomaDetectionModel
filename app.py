import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import io
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Melanoma Detection AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ABCD Feature Extractor Class (from your original code)
class ABCDFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'asymmetry_vertical', 'asymmetry_horizontal', 'asymmetry_combined',
            'border_irregularity', 'border_roughness',
            'color_mean_r', 'color_mean_g', 'color_mean_b',
            'color_std_r', 'color_std_g', 'color_std_b',
            'color_variance_total', 'color_range_r', 'color_range_g', 'color_range_b',
            'diameter_equivalent', 'area_normalized'
        ]

    def extract_features(self, image_rgb):
        """Extract ABCD features from RGB image"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        mask = self._segment_lesion(gray, image_rgb)

        if mask is None or np.sum(mask) == 0:
            return np.zeros(len(self.feature_names))

        asymmetry_features = self._calculate_asymmetry(mask)
        border_features = self._calculate_border_features(mask)
        color_features = self._calculate_color_features(image_rgb, mask)
        diameter_features = self._calculate_diameter_features(mask)

        features = np.concatenate([
            asymmetry_features,
            border_features,
            color_features,
            diameter_features
        ])

        return features

    def _segment_lesion(self, gray, image_rgb):
        """Segment lesion from background"""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        return mask

    def _calculate_asymmetry(self, mask):
        """Calculate asymmetry features"""
        h, w = mask.shape
        M = cv2.moments(mask)
        if M['m00'] == 0:
            return np.array([0, 0, 0])

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        left_area = np.sum(mask[:, :cx] == 255)
        right_area = np.sum(mask[:, cx:] == 255)
        vertical_asymmetry = abs(left_area - right_area) / (left_area + right_area + 1e-6)

        top_area = np.sum(mask[:cy, :] == 255)
        bottom_area = np.sum(mask[cy:, :] == 255)
        horizontal_asymmetry = abs(top_area - bottom_area) / (top_area + bottom_area + 1e-6)

        combined_asymmetry = (vertical_asymmetry + horizontal_asymmetry) / 2

        return np.array([vertical_asymmetry, horizontal_asymmetry, combined_asymmetry])

    def _calculate_border_features(self, mask):
        """Calculate border irregularity features"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.array([0, 0])

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        if area == 0 or perimeter == 0:
            return np.array([0, 0])

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        irregularity = 1 - circularity

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        roughness = len(contour) / len(approx) if len(approx) > 0 else 0

        return np.array([irregularity, roughness])

    def _calculate_color_features(self, image_rgb, mask):
        """Calculate color features"""
        lesion_pixels = image_rgb[mask == 255]

        if len(lesion_pixels) == 0:
            return np.zeros(10)

        color_means = np.mean(lesion_pixels, axis=0)
        color_stds = np.std(lesion_pixels, axis=0)
        color_ranges = np.ptp(lesion_pixels, axis=0)
        color_variance = np.var(lesion_pixels.flatten())

        features = np.concatenate([
            color_means,
            color_stds,
            [color_variance],
            color_ranges
        ])

        return features

    def _calculate_diameter_features(self, mask):
        """Calculate diameter features"""
        area = np.sum(mask == 255)

        if area == 0:
            return np.array([0, 0])

        equivalent_diameter = 2 * np.sqrt(area / np.pi)
        total_pixels = mask.shape[0] * mask.shape[1]
        normalized_area = area / total_pixels

        return np.array([equivalent_diameter, normalized_area])

# Load models and scaler
@st.cache_resource
def load_models():
    """Load all trained models and scaler"""
    try:
        cnn_model = load_model('models/cnn_model.h5')
        abcd_model = load_model('models/abcd_model.h5')
        combined_model = load_model('models/combined_model.h5')
        
        with open('models/abcd_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return cnn_model, abcd_model, combined_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please ensure all model files are in the 'models/' directory")
        return None, None, None, None

def preprocess_image_for_cnn(image, target_size=(224, 224)):
    """Preprocess image for CNN model"""
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def interpret_risk_level(probability):
    """Interpret model probability as risk level"""
    if probability < 0.3:
        return "Low", "low", "#4caf50"
    elif probability < 0.7:
        return "Medium", "medium", "#ff9800"
    else:
        return "High", "high", "#f44336"

def create_risk_gauge(probability, model_name):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{model_name} Risk Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70}}))
    
    fig.update_layout(height=300)
    return fig

def create_abcd_radar_chart(features, feature_names):
    """Create radar chart for ABCD features"""
    # Normalize features to 0-1 range for better visualization
    features_norm = (features - features.min()) / (features.max() - features.min() + 1e-6)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=features_norm,
        theta=feature_names,
        fill='toself',
        name='ABCD Features'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="ABCD Feature Analysis",
        height=500
    )
    
    return fig

def display_educational_content():
    """Display educational content about melanoma"""
    st.header(" Understanding Melanoma Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ABCD Rule for Melanoma")
        st.markdown("""
        **A - Asymmetry**: One half doesn't match the other half
        
        **B - Border**: Edges are irregular, ragged, notched, or blurred
        
        **C - Color**: Color is not uniform throughout
        
        **D - Diameter**: Diameter is larger than 6mm (about the size of a pencil eraser)
        """)
    
    with col2:
        st.subheader("When to See a Doctor")
        st.markdown("""
        - Any new mole or growth
        - Any mole that changes in size, shape, or color
        - Any mole that bleeds or becomes tender
        - Any suspicious skin lesion
        
        **Remember**: This AI tool is for educational purposes only and should not replace professional medical advice.
        """)

def main():
    st.markdown('<h1 class="main-header">🔬 Melanoma Detection AI</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Diagnosis", "About", "Educational Content"])
    
    if page == "About":
        st.header("About This Application")
        st.markdown("""
        This application uses advanced AI models to analyze skin lesions for potential melanoma detection.
        
        **Three Models Used:**
        1. **CNN Model**: Analyzes the visual appearance of the lesion
        2. **ABCD Model**: Evaluates specific dermatological features (Asymmetry, Border, Color, Diameter)
        3. **Combined Model**: Integrates both approaches for enhanced accuracy
        
        **Disclaimer**: This tool is for educational and research purposes only. Always consult with a qualified healthcare professional for medical diagnosis and treatment.
        """)
        return
    
    elif page == "Educational Content":
        display_educational_content()
        return
    
    # Main diagnosis page
    st.header("Upload Skin Lesion Image for Analysis")
    
    # Load models
    cnn_model, abcd_model, combined_model, scaler = load_models()
    
    if cnn_model is None:
        st.error("Models could not be loaded. Please check the model files.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the skin lesion"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Image Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size}")
            st.write(f"**Format:** {image.format}")
        
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image... This may take a moment."):
                try:
                    # Convert PIL image to numpy array
                    image_rgb = np.array(image.convert('RGB'))
                    
                    # 1. CNN Model Prediction
                    cnn_input = preprocess_image_for_cnn(image)
                    cnn_prediction = cnn_model.predict(cnn_input, verbose=0)[0][0]
                    
                    # 2. ABCD Feature Extraction and Prediction
                    extractor = ABCDFeatureExtractor()
                    abcd_features = extractor.extract_features(image_rgb)
                    abcd_features_scaled = scaler.transform([abcd_features])
                    abcd_prediction = abcd_model.predict(abcd_features_scaled, verbose=0)[0][0]
                    
                    # 3. Combined Model Prediction
                    combined_prediction = combined_model.predict([cnn_input, abcd_features_scaled], verbose=0)[0][0]
                    
                    # Display Results
                    st.header("Analysis Results")
                    
                    # Risk Assessment Cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_level, risk_class, color = interpret_risk_level(cnn_prediction)
                        st.markdown(f"""
                        <div class="risk-{risk_class}">
                            <h4>CNN Model</h4>
                            <h2 style="color: {color};">{risk_level} Risk</h2>
                            <p>Confidence: {cnn_prediction:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        risk_level, risk_class, color = interpret_risk_level(abcd_prediction)
                        st.markdown(f"""
                        <div class="risk-{risk_class}">
                            <h4>ABCD Model</h4>
                            <h2 style="color: {color};">{risk_level} Risk</h2>
                            <p>Confidence: {abcd_prediction:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        risk_level, risk_class, color = interpret_risk_level(combined_prediction)
                        st.markdown(f"""
                        <div class="risk-{risk_class}">
                            <h4>Combined Model</h4>
                            <h2 style="color: {color};">{risk_level} Risk</h2>
                            <p>Confidence: {combined_prediction:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk Gauges
                    st.subheader("Detailed Risk Assessment")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        fig_cnn = create_risk_gauge(cnn_prediction, "CNN")
                        st.plotly_chart(fig_cnn, use_container_width=True)
                    
                    with col2:
                        fig_abcd = create_risk_gauge(abcd_prediction, "ABCD")
                        st.plotly_chart(fig_abcd, use_container_width=True)
                    
                    with col3:
                        fig_combined = create_risk_gauge(combined_prediction, "Combined")
                        st.plotly_chart(fig_combined, use_container_width=True)
                    
                    # ABCD Feature Analysis
                    st.subheader("🔍 ABCD Feature Analysis")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Radar chart
                        fig_radar = create_abcd_radar_chart(abcd_features, extractor.feature_names)
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    with col2:
                        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
                        st.subheader("Feature Values")
                        
                        # Group features by category
                        feature_groups = {
                            "Asymmetry": abcd_features[:3],
                            "Border": abcd_features[3:5],
                            "Color": abcd_features[5:15],
                            "Diameter": abcd_features[15:17]
                        }
                        
                        for group, values in feature_groups.items():
                            st.write(f"**{group}**: {np.mean(values):.3f}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Feature Details Table
                    st.subheader("📋 Detailed Feature Analysis")
                    
                    feature_df = pd.DataFrame({
                        'Feature': extractor.feature_names,
                        'Value': abcd_features,
                        'Normalized': abcd_features_scaled[0]
                    })
                    
                    st.dataframe(feature_df, use_container_width=True)
                    
                    # Clinical Recommendations
                    st.subheader("🏥 Clinical Recommendations")
                    
                    max_risk = max(cnn_prediction, abcd_prediction, combined_prediction)
                    
                    if max_risk < 0.3:
                        st.success("""
                        **Low Risk Assessment**
                        - Continue regular self-examinations
                        - Monitor for any changes in size, shape, or color
                        - Consider annual dermatological check-ups
                        """)
                    elif max_risk < 0.7:
                        st.warning("""
                        **Medium Risk Assessment**
                        - Schedule an appointment with a dermatologist within 2-4 weeks
                        - Monitor the lesion closely for any changes
                        - Take photos to track changes over time
                        - Avoid sun exposure to the area
                        """)
                    else:
                        st.error("""
                        **High Risk Assessment**
                        - **Urgent**: Schedule an appointment with a dermatologist immediately
                        - Consider seeking a second opinion from a dermatology specialist
                        - Avoid disturbing the lesion
                        - Document any changes with photographs
                        - This requires prompt medical evaluation
                        """)
                    
                    # Download Results
                    st.subheader("Download Results")
                    
                    results_dict = {
                        'CNN_Prediction': float(cnn_prediction),
                        'ABCD_Prediction': float(abcd_prediction),
                        'Combined_Prediction': float(combined_prediction),
                        'ABCD_Features': abcd_features.tolist(),
                        'Feature_Names': extractor.feature_names,
                        'Timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    results_json = pd.Series(results_dict).to_json()
                    
                    st.download_button(
                        label="Download Analysis Results (JSON)",
                        data=results_json,
                        file_name=f"melanoma_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.error("Please try with a different image or check that the image is clear and properly formatted.")

if __name__ == "__main__":
    main()