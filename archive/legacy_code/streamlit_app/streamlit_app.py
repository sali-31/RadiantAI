"""
LesionRec - Advanced Acne Detection Dashboard

Multi-model AI system combining YOLOv8, YOLOv10, and Gemini Vision
for comprehensive acne analysis and treatment recommendations.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path
import json
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO
import os
import torch

# Fix PyTorch 2.6+ weights_only issue with YOLO models
# Add safe globals for Ultralytics classes
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except:
    pass

# Also patch torch.load as fallback
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for YOLO models"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from scripts.product_recommendations import ProductRecommendationSystem

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

try:
    import google.generativeai as genai
    from scripts.gemini_analysis import GeminiAcneAnalyzer
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="LesionRec - AI Acne Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_yolov8_model():
    """Load YOLOv8 model"""
    model_path = project_root / "runs" / "detect" / "acne_yolov8_production" / "weights" / "best.pt"

    if not model_path.exists():
        return None

    try:
        # PyTorch 2.6+ requires weights_only=False for custom models
        return YOLO(str(model_path))
    except Exception as e:
        st.error(f"Error loading YOLOv8: {e}")
        return None


@st.cache_resource
def load_yolov10_model():
    """Load YOLOv10 model"""
    model_path = project_root / "runs" / "detect" / "acne_yolov10_production" / "weights" / "best.pt"

    if not model_path.exists():
        return None

    try:
        from ultralytics import YOLOv10
        return YOLOv10(str(model_path))
    except:
        return None


@st.cache_resource
def load_gemini_analyzer():
    """Load Gemini analyzer"""
    api_key = os.environ.get('GEMINI_API_KEY')

    if not api_key:
        return None

    try:
        return GeminiAcneAnalyzer(api_key=api_key)
    except:
        return None


@st.cache_resource
def load_product_recommender():
    """Load product recommendation system"""
    try:
        return ProductRecommendationSystem()
    except:
        return None


def detect_with_yolo(model, image, model_name="YOLO"):
    """Run YOLO detection on image"""
    if model is None:
        return None

    # Run inference
    results = model.predict(image, conf=0.25, iou=0.45, verbose=False)

    if not results or len(results) == 0:
        return None

    result = results[0]

    # Extract detections
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            'total_detections': 0,
            'comedones_count': 0,
            'papules_count': 0,
            'pustules_count': 0,
            'nodules_count': 0,
            'annotated_image': image
        }

    # Count by class
    class_names = ['comedone', 'papule', 'pustule', 'nodule']
    counts = {f'{name}s_count': 0 for name in class_names}

    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id < len(class_names):
            counts[f'{class_names[cls_id]}s_count'] += 1

    # Get annotated image
    annotated = result.plot()  # Returns BGR numpy array
    annotated_rgb = Image.fromarray(annotated[..., ::-1])  # Convert BGR to RGB

    return {
        'total_detections': len(boxes),
        **counts,
        'annotated_image': annotated_rgb,
        'confidence_scores': [float(box.conf[0]) for box in boxes]
    }


def analyze_with_gemini(analyzer, image_path):
    """Run Gemini analysis"""
    if analyzer is None:
        return None

    try:
        result = analyzer.analyze_acne(image_path, detailed=False)
        return result
    except Exception as e:
        st.error(f"Gemini analysis failed: {e}")
        return None


def display_detection_results(results, model_name):
    """Display detection results in nice format"""
    if results is None:
        st.warning(f"{model_name} model not available")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(results['annotated_image'], caption=f"{model_name} Detection", use_column_width=True)

    with col2:
        st.metric("Total Lesions", results['total_detections'])

        st.write("**Breakdown:**")
        st.write(f"🔴 Comedones: {results['comedones_count']}")
        st.write(f"🟠 Papules: {results['papules_count']}")
        st.write(f"🟡 Pustules: {results['pustules_count']}")
        st.write(f"🔵 Nodules: {results['nodules_count']}")

        if results['confidence_scores']:
            avg_conf = np.mean(results['confidence_scores']) * 100
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")


def display_gemini_results(results):
    """Display Gemini analysis results"""
    if results is None:
        st.warning("Gemini analysis not available")
        return

    st.subheader("Gemini Vision Analysis")

    col1, col2 = st.columns(2)

    with col1:
        severity = results.get('severity', 'unknown')
        severity_color = {
            'mild': '🟢',
            'moderate': '🟡',
            'severe': '🔴'
        }.get(severity, '⚪')

        st.metric("Severity", f"{severity_color} {severity.upper()}")

        if 'estimated_count' in results:
            st.metric("Estimated Lesions", results['estimated_count'])

    with col2:
        if 'skin_type' in results:
            st.write(f"**Skin Type:** {results['skin_type']}")

        if 'lesion_types' in results:
            st.write("**Lesion Types:**")
            for lesion_type in results['lesion_types']:
                st.write(f"  • {lesion_type}")

    if 'summary' in results:
        st.info(results['summary'])

    if 'concerns' in results:
        st.write("**Concerns:**")
        for concern in results['concerns']:
            st.write(f"  ⚠️ {concern}")

    if 'recommendations' in results:
        with st.expander("💡 Gemini Recommendations"):
            for rec in results['recommendations']:
                st.write(f"• {rec}")


def display_product_recommendations(recommender, detection_results):
    """Display product recommendations"""
    if recommender is None:
        st.warning("Product recommendation system not available")
        return

    st.subheader("🛍️ Personalized Product Recommendations")

    # Get recommendations
    recommendations = recommender.recommend(detection_results, budget="moderate")

    # Analysis summary
    analysis = recommendations['analysis']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Severity", analysis['severity'].upper())

    with col2:
        st.metric("Dominant Type", analysis['dominant_type'].upper())

    with col3:
        st.metric("Total Lesions", analysis['total_lesions'])

    # Products
    products = recommendations['products']

    st.write("---")

    # Cleanser
    if 'cleanser' in products and products['cleanser']:
        with st.expander("🧼 **CLEANSER**", expanded=True):
            for p in products['cleanser']:
                st.write(f"**{p['name']}**")
                st.write(f"💊 {p.get('key_ingredient', 'N/A')}")
                st.write(f"💵 {p.get('price_range', 'N/A')}")
                st.write(f"✨ {p.get('why_it_works', 'N/A')}")
                if 'how_to_use' in p:
                    st.write(f"📝 {p['how_to_use']}")

    # Treatment
    if 'treatment' in products and products['treatment']:
        with st.expander("💊 **TREATMENT**", expanded=True):
            for i, p in enumerate(products['treatment']):
                if i > 0:
                    st.write("---")
                st.write(f"**{p['name']}**")
                st.write(f"💊 {p.get('key_ingredient', 'N/A')}")
                st.write(f"💵 {p.get('price_range', 'N/A')}")
                st.write(f"✨ {p.get('why_it_works', 'N/A')}")
                if 'how_to_use' in p:
                    st.write(f"📝 {p['how_to_use']}")

    # Moisturizer
    if 'moisturizer' in products and products['moisturizer']:
        with st.expander("💧 **MOISTURIZER**", expanded=True):
            for p in products['moisturizer']:
                st.write(f"**{p['name']}**")
                st.write(f"💊 {p.get('key_ingredient', 'N/A')}")
                st.write(f"💵 {p.get('price_range', 'N/A')}")
                st.write(f"✨ {p.get('why_it_works', 'N/A')}")

    # Routine
    routine = recommendations.get('routine', {})
    if routine:
        with st.expander("📅 **DAILY ROUTINE**"):
            st.write(f"**{routine.get('summary', '')}**")

            if 'morning' in routine:
                st.write("\n**Morning:**")
                for step in routine['morning']:
                    st.write(step)

            if 'evening' in routine:
                st.write("\n**Evening:**")
                for step in routine['evening']:
                    st.write(step)

    # Timeline & Tips
    col1, col2 = st.columns(2)

    with col1:
        if 'timeline' in recommendations:
            st.info(f"⏱️ **Timeline:** {recommendations['timeline']}")

    with col2:
        if 'pro_tip' in recommendations:
            st.success(f"💡 **Pro Tip:** {recommendations['pro_tip']}")

    if 'when_to_see_doctor' in recommendations:
        st.warning(f"⚕️ **When to see a doctor:** {recommendations['when_to_see_doctor']}")


def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 LesionRec</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced AI-Powered Acne Detection & Treatment System</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        st.write("**Available Models:**")
        yolov8_model = load_yolov8_model()
        st.write(f"{'✅' if yolov8_model else '❌'} YOLOv8")

        yolov10_model = load_yolov10_model()
        st.write(f"{'✅' if yolov10_model else '❌'} YOLOv10")

        gemini_analyzer = load_gemini_analyzer()
        st.write(f"{'✅' if gemini_analyzer else '❌'} Gemini Vision")

        product_recommender = load_product_recommender()
        st.write(f"{'✅' if product_recommender else '❌'} Product Recommendations")

        st.write("---")

        # Model selection
        models_to_run = st.multiselect(
            "Select models to run:",
            ["YOLOv8", "YOLOv10", "Gemini Vision"],
            default=["YOLOv8"]
        )

        show_products = st.checkbox("Show product recommendations", value=True)

        st.write("---")
        st.caption("© 2025 LesionRec | Powered by AI")

    # Main content
    st.write("## Upload Skin Image")

    uploaded_file = st.file_uploader(
        "Choose a skin image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of the affected skin area"
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)

        # Save temporarily for Gemini
        temp_path = project_root / "temp_upload.jpg"
        image.save(temp_path)

        # Display original
        st.write("---")
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", width=400)

        # Analyze button
        if st.button("🔍 Analyze Skin", type="primary"):
            st.write("---")
            st.subheader("Analysis Results")

            # Results containers
            yolov8_results = None
            yolov10_results = None
            gemini_results = None

            # Run selected models
            with st.spinner("Analyzing image..."):
                if "YOLOv8" in models_to_run and yolov8_model:
                    with st.status("Running YOLOv8 detection..."):
                        yolov8_results = detect_with_yolo(yolov8_model, image, "YOLOv8")

                if "YOLOv10" in models_to_run and yolov10_model:
                    with st.status("Running YOLOv10 detection..."):
                        yolov10_results = detect_with_yolo(yolov10_model, image, "YOLOv10")

                if "Gemini Vision" in models_to_run and gemini_analyzer:
                    with st.status("Running Gemini Vision analysis..."):
                        gemini_results = analyze_with_gemini(gemini_analyzer, str(temp_path))

            # Display results in tabs
            tabs = []
            if yolov8_results:
                tabs.append("YOLOv8")
            if yolov10_results:
                tabs.append("YOLOv10")
            if gemini_results:
                tabs.append("Gemini Vision")

            if tabs:
                tab_objects = st.tabs(tabs)

                for i, tab_name in enumerate(tabs):
                    with tab_objects[i]:
                        if tab_name == "YOLOv8":
                            display_detection_results(yolov8_results, "YOLOv8")
                        elif tab_name == "YOLOv10":
                            display_detection_results(yolov10_results, "YOLOv10")
                        elif tab_name == "Gemini Vision":
                            display_gemini_results(gemini_results)

            # Model comparison
            if yolov8_results and yolov10_results:
                st.write("---")
                st.subheader("📊 Model Comparison")

                comp_data = {
                    'Model': ['YOLOv8', 'YOLOv10'],
                    'Total Detections': [
                        yolov8_results['total_detections'],
                        yolov10_results['total_detections']
                    ],
                    'Avg Confidence': [
                        f"{np.mean(yolov8_results['confidence_scores'])*100:.1f}%" if yolov8_results['confidence_scores'] else 'N/A',
                        f"{np.mean(yolov10_results['confidence_scores'])*100:.1f}%" if yolov10_results['confidence_scores'] else 'N/A'
                    ]
                }

                st.dataframe(comp_data, use_container_width=True)

            # Product recommendations
            if show_products and product_recommender:
                # Use YOLOv8 results if available, otherwise YOLOv10
                detection_for_products = yolov8_results or yolov10_results

                if detection_for_products:
                    st.write("---")
                    display_product_recommendations(product_recommender, detection_for_products)

            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    else:
        # Instructions
        st.info("""
### How to use:
1. Upload a clear photo of your skin
2. Select which AI models to run (sidebar)
3. Click "Analyze Skin"
4. Review results from multiple AI models
5. Get personalized product recommendations

### Supported Models:
- **YOLOv8**: Fast, accurate lesion detection
- **YOLOv10**: Latest YOLO with improved small object detection
- **Gemini Vision**: Natural language analysis and insights
        """)

        # Sample images
        st.write("### Sample Images")
        st.caption("Try uploading a sample acne image to test the system")


if __name__ == "__main__":
    main()
