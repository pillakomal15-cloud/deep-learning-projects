import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import os

# Page configuration
st.set_page_config(
    page_title="TATTVA AI | Digit Recognition",
    page_icon="✍️",
    layout="wide"
)

# Professional styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f, #101b30);
    color: #f8fafc;
}
h1, h2, h3, p, label {
    color: #f8fafc;
}
.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(120deg, #14243d, #102b38);
    border: 1px solid #28516a;
    margin-bottom: 25px;
}
.result {
    padding: 25px;
    border-radius: 15px;
    background: #102b38;
    border: 1px solid #2dd4bf;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero">
    <h1>✍️ TATTVA AI</h1>
    <h3>Handwritten Digit Recognition System</h3>
    <p>Upload a handwritten digit image and let a trained neural
    network predict the number.</p>
</div>
""", unsafe_allow_html=True)

# Load model from the repository root
@st.cache_resource
def load_digit_model():
    model_path = "mnist_model.h5"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "mnist_model.h5 was not found in the repository root."
        )

    return tf.keras.models.load_model(model_path, compile=False)


# Model loading
try:
    model = load_digit_model()
    st.success("AI model loaded successfully.")
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.info(
        "Check that mnist_model.h5 exists in the same directory "
        "as app.py in GitHub."
    )
    st.stop()

# Main layout
left, right = st.columns([1, 1])

with left:
    st.subheader("📤 Upload Digit Image")

    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG, or JPEG image",
        type=["png", "jpg", "jpeg"]
    )

    st.caption(
        "Tip: Upload a clear image containing one handwritten digit."
    )

    predict_button = st.button(
        "🔍 Recognize Digit",
        type="primary",
        use_container_width=True
    )

with right:
    st.subheader("🤖 Prediction Result")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if predict_button:
            # Convert to grayscale
            gray = ImageOps.grayscale(image)

            # Resize to MNIST dimensions
            gray = ImageOps.fit(
                gray,
                (28, 28),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5)
            )

            # MNIST models usually expect light digits on a dark background.
            # Invert when the image background is predominantly light.
            arr = np.asarray(gray, dtype=np.float32)

            if arr.mean() > 127:
                arr = 255.0 - arr

            # Normalize pixels to 0–1
            arr = arr / 255.0

            # Shape: batch, height, width
            input_data = arr.reshape(1, 28, 28)

            # Add channel dimension for CNN models
            if len(model.input_shape) == 4:
                input_data = input_data[..., np.newaxis]

            # Prediction
            with st.spinner("Analyzing your digit..."):
                prediction = model.predict(input_data, verbose=0)
                probabilities = prediction[0]

                digit = int(np.argmax(probabilities))
                confidence = float(np.max(probabilities)) * 100

            st.markdown(
                f"""
                <div class="result">
                    <p>Predicted Digit</p>
                    <h1 style="font-size:80px;color:#2dd4bf;">
                        {digit}
                    </h1>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("📊 Prediction Probabilities")

            chart_data = {
                str(i): float(probabilities[i])
                for i in range(len(probabilities))
            }

            st.bar_chart(chart_data)

    else:
        st.info("Upload a digit image to begin.")

st.divider()

st.markdown(
    "<center>© 2026 TATTVA AI | Deep Learning Project</center>",
    unsafe_allow_html=True
)
