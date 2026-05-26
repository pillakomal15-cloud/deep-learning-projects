import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model/mnist_model.h5")

# App title
st.title("Handwritten Digit Recognition")

st.write("Draw a digit below and click Predict")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict"):

    if canvas_result.image_data is not None:

        # Convert image to grayscale
        img = Image.fromarray(
            (canvas_result.image_data[:, :, 0]).astype(np.uint8)
        )

        # Resize image
        img = img.resize((28, 28))

        # Convert image to array
        img_array = np.array(img)

        # Normalize
        img_array = img_array / 255.0

        # Reshape for CNN
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)

        predicted_digit = np.argmax(prediction)

        # Show result
        st.success(f"Predicted Digit: {predicted_digit}")