import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model_pruned.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print("Input details:", input_details)