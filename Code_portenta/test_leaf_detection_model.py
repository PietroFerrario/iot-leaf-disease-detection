import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model 
interpreter = tf.lite.Interpreter(model_path="leaf_detection.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#  Print model info 
print("Input:", input_details)
print("Output:", output_details)

#  Load and preprocess the image 
def load_image(path):
    img = Image.open(path).convert('L')  # Grayscale
    img = img.resize((96, 96))           # Resize to match model input
    img_np = np.array(img, dtype=np.uint8) 
    
    img_np = (img_np -128).astype(np.int8)


    img_np = np.expand_dims(img_np, axis=(0, -1))  # Shape: (1, 96, 96, 1)
    return img_np

image = load_image("test_image.png")

#  Set tensor input 
interpreter.set_tensor(input_details[0]['index'], image)

#Run inference
interpreter.invoke()

#  Get prediction 
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Raw output:", output_data)

# Get predicted class 
predicted_class = np.argmax(output_data)
print("Predicted class:", predicted_class)
