import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model_pruned.tflite")

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Generate a random input tensor with the same shape as the model input
input_shape = input_details[0]['shape']
test_input = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], test_input)

# Invoke the interpreter to run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Model output:", output_data)

# Check the output shape and data type
print("Output shape:", output_data.shape)
print("Output type:", output_data.dtype)
