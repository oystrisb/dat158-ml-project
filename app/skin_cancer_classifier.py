import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model/final_skin_cancer_model.keras")


# Define a prediction function
def classify_mole(image):
    image = cv2.resize(image, (128, 128))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = model.predict(image_array)[0][0]
    result = "Malignant" if prediction > 0.5 else "Benign"
    confidence = f"{(prediction * 100):.2f}%" if result == "Malignant" else f"{((1 - prediction) * 100):.2f}%"
    return f"{result} ({confidence})"

image = gr.components.Image() 

# Define the Gradio interface
interface = gr.Interface(
    fn=classify_mole,
    inputs=image,
    outputs="text",
    title="Mole Scanner",
    description="Upload an image of a mole to classify it as benign or malignant."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()

