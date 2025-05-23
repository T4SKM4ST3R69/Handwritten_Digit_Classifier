import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import joblib
from PIL import Image, ImageOps
from scipy import ndimage

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        clf = joblib.load('logistic_regression_mnist_model.joblib')
        scaler = joblib.load('mnist_scaler.joblib')
        return clf, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please run the training script first.")
        return None, None

def preprocess_canvas_image_no_cv2(canvas_data):
    """
    Preprocessing without OpenCV - using PIL and scipy instead
    """
    if canvas_data is None:
        return None
    
    # Convert to PIL Image
    img = Image.fromarray(canvas_data.astype('uint8'), 'RGBA')
    
    # Convert to grayscale
    img_gray = img.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img_gray)
    
    # Find bounding box using numpy operations
    rows = np.any(img_array, axis=1)
    cols = np.any(img_array, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding
        padding = 20
        rmin = max(0, rmin - padding)
        rmax = min(img_array.shape[0], rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(img_array.shape[1], cmax + padding)
        
        # Crop to bounding box
        img_cropped = img_array[rmin:rmax, cmin:cmax]
    else:
        img_cropped = img_array
    
    # Resize to 20x20 first
    img_pil = Image.fromarray(img_cropped)
    img_20x20 = img_pil.resize((20, 20), Image.Resampling.LANCZOS)
    
    # Create 28x28 image with 4-pixel border
    img_28x28 = Image.new('L', (28, 28), 0)
    img_28x28.paste(img_20x20, (4, 4))
    
    # Convert to numpy array
    final_array = np.array(img_28x28)
    
    # Normalize to 0-1 range
    final_array = final_array.astype('float64') / 255.0
    
    return final_array

clf, scaler = load_model()

st.title('🔢 MNIST Digit Recognition')
st.markdown("### Draw a digit (0-9) in the canvas below")

if clf is not None and scaler is not None:
    # Create columns for better layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**Drawing Canvas:**")
        canvas_result = st_canvas(
            fill_color='#000000',
            stroke_width=12,
            stroke_color='#FFFFFF',
            background_color='#000000',
            height=280,
            width=280,
            drawing_mode='freedraw',
            key='canvas'
        )
    
    with col2:
        st.markdown("**Tips for better accuracy:**")
        st.markdown("- Draw digits **large and centered**")
        st.markdown("- Use **bold strokes**")
        st.markdown("- Keep digits **well-formed**")
        st.markdown("- Avoid touching canvas edges")
        
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.rerun()
    
    # Process and predict
    if canvas_result.image_data is not None:
        # Check if something is drawn
        if np.any(canvas_result.image_data[:, :, 3] > 0):
            
            # Preprocess the image (without OpenCV)
            processed_img = preprocess_canvas_image_no_cv2(canvas_result.image_data)
            
            if processed_img is not None:
                # Display processed image
                st.markdown("---")
                col3, col4 = st.columns([1, 2])
                
                with col3:
                    st.markdown("**Processed Image (28×28):**")
                    st.image(processed_img, width=140, caption="Ready for prediction")
                
                with col4:
                    # Flatten and scale for prediction
                    img_flat = processed_img.flatten().reshape(1, -1)
                    img_scaled = scaler.transform(img_flat)
                    
                    # Get prediction and probabilities
                    prediction = clf.predict(img_scaled)[0]
                    probabilities = clf.predict_proba(img_scaled)[0]
                    confidence = np.max(probabilities)
                    
                    # Display results
                    st.markdown("**Prediction Results:**")
                    st.markdown(f"## 🎯 **{prediction}**")
                    st.markdown(f"**Confidence: {confidence:.1%}**")
                    
                    # Show confidence bar
                    st.progress(confidence)
                
                # Detailed probability breakdown
                st.markdown("**All Probabilities:**")
                import pandas as pd
                prob_data = {f"Digit {i}": prob for i, prob in enumerate(probabilities)}
                df = pd.DataFrame(list(prob_data.items()), columns=['Digit', 'Probability'])
                
                st.bar_chart(df.set_index('Digit')['Probability'])
                
                # Additional feedback
                if confidence < 0.7:
                    st.warning("⚠️ Low confidence prediction. Try redrawing the digit more clearly.")
                elif confidence > 0.95:
                    st.success("✅ High confidence prediction!")
        
        else:
            st.info("👆 Draw a digit on the canvas above to see the prediction!")

else:
    st.error("❌ Model files not found!")
