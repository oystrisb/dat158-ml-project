import gradio as gr
from gradio import components
import tensorflow as tf
#import numpy as np
import cv2
#from PIL import Image

# Load the model
loaded_model = tf.keras.models.load_model('model/final_skin_cancer_model.h5', compile=False)  # Load the model
loaded_model.compile(optimizer='adam',
              loss='categorical_crossentropy',  
              metrics=['accuracy'])

labels = ['Benign', 'Malignant']

def classify_mole(img):
 
  # Resize the image (224x224 pixels) and preprocess it
  img = cv2.resize(img, (224, 224))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)
  prediction = loaded_model.predict(img_array)

  # Get the predicted label
  score = tf.nn.softmax(prediction[0])
  label = labels[tf.argmax(score)]
  return label


# Defining gradio components
image = gr.components.Image()  
label = gr.components.Label()  

# Create a Gradio interface
iface = gr.Interface(
    fn=classify_mole, 
    inputs=image,       
    outputs=label,       
)

# Launch the Gradio interface
iface.launch(share=True)
